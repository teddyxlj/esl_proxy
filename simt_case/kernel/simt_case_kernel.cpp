#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>
#include "cce_aicore_intrinsics.h"

#include "../common/simt_case_protocol.h"

#ifndef SIMT_CASE_WARP_COUNT
#define SIMT_CASE_WARP_COUNT 8
#endif

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpCount = SIMT_CASE_WARP_COUNT;
constexpr uint32_t kThreadCount = kWarpCount * kWarpSize;
constexpr int kSingleCacheLine = 0;

/* ── Scalar helpers (callable from AIC/AIV Scalar entry, NOT from VF) ── */

__aicore__ inline uint64_t ScalarAtomicLoad(__gm__ volatile int64_t *addr) {
    return (uint64_t)atomicAdd((__gm__ int64_t *)addr, (int64_t)0);
}

__aicore__ inline uint64_t ScalarCAS(__gm__ volatile int64_t *addr, uint64_t expected, uint64_t desired) {
    return (uint64_t)atomicCAS((__gm__ int64_t *)addr, (int64_t)expected, (int64_t)desired);
}

__aicore__ inline uint64_t ScalarFetchAdd(__gm__ volatile int64_t *addr, int64_t inc) {
    return (uint64_t)atomicAdd((__gm__ int64_t *)addr, inc);
}

__aicore__ inline void QueueLock(__gm__ GmQueue *q) {
    while (ScalarCAS(&q->lock, 0, 1) != 0) {}
}

__aicore__ inline void QueueUnlock(__gm__ GmQueue *q) {
    (void)ScalarCAS(&q->lock, 1, 0);
}

__aicore__ inline bool QueueEnqueue(__gm__ GmQueue *q, uint32_t task_id) {
    QueueLock(q);
    if (q->cnt >= SIMT_CASE_RING_SIZE) {
        QueueUnlock(q);
        return false;
    }
    q->tasks[q->tail % SIMT_CASE_RING_SIZE] = task_id;
    q->tail++;
    q->cnt++;
    QueueUnlock(q);
    return true;
}

__aicore__ inline bool QueueDequeue(__gm__ GmQueue *q, uint32_t *out) {
    QueueLock(q);
    if (q->cnt == 0) {
        QueueUnlock(q);
        return false;
    }
    *out = q->tasks[q->head % SIMT_CASE_RING_SIZE];
    q->head++;
    q->cnt--;
    QueueUnlock(q);
    return true;
}

/* ── SIMT VF: desc phase (warp-interleaved write) ── */

#if defined(__DAV_VEC__)
extern "C" __simd_vf__ __aicore__ void simt_case_simd_anchor(__ubuf__ uint32_t *scratch) {
    scratch[0] = scratch[0] + 1U;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void
SimtDescVF(
    __gm__ uint64_t *state_words,
    uint64_t total_task_cnt,
    uint64_t warp_count
) {
    const uint32_t thread = (uint32_t)threadIdx.x;
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;

    if (warp >= (uint32_t)warp_count) return;

    __gm__ SimtCaseState *state = (__gm__ SimtCaseState *)state_words;
    __gm__ volatile int64_t *pred_ring_cursor = &state->pred_ring_cursor.value;

    for (uint32_t task_id = warp; task_id < (uint32_t)total_task_cnt; task_id += (uint32_t)warp_count) {
        __gm__ GmTaskDesc *desc = &state->basic_buf[task_id];
        desc->id = task_id;
        desc->type = (task_id % 3 == 0) ? (uint32_t)TASK_TYPE_CUBE : (uint32_t)TASK_TYPE_VECTOR;
        desc->count = 1;
        desc->duration = 100 + (task_id % 50);
        desc->tensor_cnt = 0;
        desc->scalar_cnt = 0;

        if (task_id > 0) {
            uint64_t old = asc_atomic_add(
                (__gm__ uint64_t *)pred_ring_cursor, 1);
            uint32_t pred_offset = (uint32_t)old;
            state->pred_ring[pred_offset] = task_id - 1;
            state->predecessors[task_id].cnt = 1;
            state->predecessors[task_id].exp_offset = pred_offset;
        } else {
            state->predecessors[task_id].cnt = 0;
            state->predecessors[task_id].exp_offset = 0;
        }

        state->state_buf[task_id].state = TASK_STATE_CREATING;
        state->state_buf[task_id].successor_cnt = 0;
        state->successor_buf[task_id].cnt = 0;

        if (lane == 0) {
            asc_atomic_add((__gm__ uint64_t *)&state->report_desc_writes, 1);
        }
    }
}
#endif

/* ── Scalar: cutter phase ── */

__aicore__ inline void RunCutter(__gm__ SimtCaseState *state) {
    uint32_t total = state->total_task_cnt;

    for (uint32_t i = 0; i < total; i++) {
        if (state->predecessors[i].cnt == 0) {
            uint32_t type = state->basic_buf[i].type;
            if (type >= 2) type = TASK_TYPE_VECTOR;
            QueueEnqueue(&state->ready_queue[type], i);
            state->predecessor_cnt[i] = 0;
        } else {
            uint32_t pending = 0;
            for (uint32_t j = 0; j < state->predecessors[i].cnt; j++) {
                uint32_t pred = state->pred_ring[state->predecessors[i].exp_offset + j];
                if (state->state_buf[pred].state != TASK_STATE_COMPLETED) {
                    uint32_t sidx = state->successor_buf[pred].cnt;
                    if (sidx < SIMT_CASE_CON_NODE_CNT) {
                        state->successor_buf[pred].node[sidx] = i;
                    }
                    state->successor_buf[pred].cnt++;
                    state->state_buf[pred].successor_cnt++;
                    pending++;
                }
            }
            state->predecessor_cnt[i] = pending;
            if (pending == 0) {
                uint32_t type = state->basic_buf[i].type;
                if (type >= 2) type = TASK_TYPE_VECTOR;
                QueueEnqueue(&state->ready_queue[type], i);
            }
        }
    }
    state->report_cutter_ready = total;
}

/* ── Scalar: dispatch + immediate-complete executor ── */

__aicore__ inline void RunDispatchAndExecute(__gm__ SimtCaseState *state) {
    uint32_t sent = 0;
    uint32_t done = 0;

    for (uint32_t type = 0; type < 2; type++) {
        uint32_t task_id;
        while (QueueDequeue(&state->ready_queue[type], &task_id)) {
            state->state_buf[task_id].state = TASK_STATE_COMPLETED;
            done++;

            uint32_t sc = state->state_buf[task_id].successor_cnt;
            for (uint32_t k = 0; k < sc; k++) {
                uint32_t succ = state->successor_buf[task_id].node[k];
                uint32_t prev = (uint32_t)ScalarFetchAdd(
                    (__gm__ volatile int64_t *)&state->predecessor_cnt[succ], -1);
                if (prev == 1) {
                    uint32_t stype = state->basic_buf[succ].type;
                    if (stype >= 2) stype = TASK_TYPE_VECTOR;
                    QueueEnqueue(&state->ready_queue[stype], succ);
                }
            }
            sent++;
        }
    }
    state->report_dispatch_sent = sent;
    state->report_executor_done = done;
}

} 

/* ── AIV build: VF + AIV entry (scheduler core) ── */
#if defined(__DAV_VEC__)

/* ── SIMD anchor: forces MIX classification (SIMD+SIMT) ── */
static __aicore__ __attribute__((noinline, used)) void SimdForceMix(__gm__ float *ga_ptr, __gm__ float *gc_ptr, uint32_t repeats) {
    constexpr int kTile = 128;
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, kTile, kTile>,
        pto::Stride<kTile * kTile, kTile * kTile, kTile * kTile, kTile, 1>>;
    using VecTile = Tile<TileType::Vec, float, kTile, kTile, BLayout::RowMajor, -1, -1>;
    GlobalData ga(ga_ptr);
    GlobalData gc(gc_ptr);
    VecTile a, b, c;
    TASSIGN(a, 0x0); TASSIGN(b, 0x10000); TASSIGN(c, 0x20000);
    for (uint32_t i = 0; i < repeats; ++i) {
        TLOAD(a, ga); TLOAD(b, ga);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(c, a, b);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(gc, c);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aiv(__gm__ SimtCaseState *state) {
    dcci((__gm__ void *)&state->phase, kSingleCacheLine);
    dsb(DSB_ALL);

    const uint32_t block = (uint32_t)get_block_idx();
    const uint32_t subblock_dim = (uint32_t)get_subblockdim();
    const uint32_t subblock = (uint32_t)get_subblockid();
    const uint32_t aiv_id = block * subblock_dim + subblock;

    if (aiv_id != 0) return;

    SimdForceMix((__gm__ float *)&state->simd_workspace[0],
                 (__gm__ float *)&state->simd_workspace[32768],
                 state->total_task_cnt > 0 ? 1u : 0u);

    state->report_kernel_start_tick = (uint64_t)get_sys_cnt();

    cce::async_invoke<SimtDescVF>(
        cce::dim3{kThreadCount, 1U, 1U},
        (__gm__ uint64_t *)state,
        (uint64_t)state->total_task_cnt,
        (uint64_t)kWarpCount
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    RunCutter(state);
    RunDispatchAndExecute(state);

    state->report_kernel_end_tick = (uint64_t)get_sys_cnt();
    (void)ScalarCAS(&state->all_done.value, 0, 1);
}

/* ── AIC build: AIC entry (executor cores, poll for all_done) ── */
#else

static __aicore__ __attribute__((noinline, used)) void SimdForceMixAic(__gm__ float *a, __gm__ float *b, __gm__ float *c, uint32_t repeats) {
    constexpr int kTile = 128;
    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kTile, kTile>,
        pto::Stride<kTile * kTile, kTile * kTile, kTile * kTile, kTile, 1>>;
    using TileMatA = Tile<TileType::Mat, float, kTile, kTile, BLayout::ColMajor, kTile, kTile, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, kTile, kTile, BLayout::ColMajor, kTile, kTile, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, kTile, kTile, kTile, kTile>;
    using RightTile = TileRight<float, kTile, kTile, kTile, kTile>;
    using AccTile = TileAcc<float, kTile, kTile, kTile, kTile>;

    GlobalData ga(a), gb(b), gc(c);
    TileMatA ma; TileMatB mb;
    LeftTile la; RightTile lb; AccTile oc;
    TASSIGN(ma, 0x0); TASSIGN(mb, 0x10000);
    TASSIGN(la, 0x0); TASSIGN(lb, 0x0); TASSIGN(oc, 0x0);
    for (uint32_t i = 0; i < repeats; ++i) {
        TLOAD(ma, ga); TLOAD(mb, gb);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(la, ma); TMOV(lb, mb);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(oc, la, lb);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE(gc, oc);
        set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    }
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aic(__gm__ SimtCaseState *state) {
    dcci((__gm__ void *)&state->phase, kSingleCacheLine);
    dsb(DSB_ALL);

    SimdForceMixAic((__gm__ float *)&state->simd_workspace[0],
                    (__gm__ float *)&state->simd_workspace[32768],
                    (__gm__ float *)&state->simd_workspace[49152],
                    state->total_task_cnt > 0 ? 1u : 0u);

    const uint32_t owner = (uint32_t)get_block_idx();
    (void)owner;

    while (ScalarAtomicLoad(&state->all_done.value) == 0) {}

    (void)ScalarFetchAdd((__gm__ volatile int64_t *)&state->executor_done[owner % 64].value, 1);
}

#endif
