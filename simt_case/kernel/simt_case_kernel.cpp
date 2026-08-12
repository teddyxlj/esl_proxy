#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>
#include "cce_aicore_intrinsics.h"

#include "../common/simt_case_protocol.h"

#ifndef SC_WARP_COUNT
#define SC_WARP_COUNT 8
#endif

/* ── SIMT VF: desc + cutter + dispatch (all in one VF, warp-split) ── */

#if defined(__DAV_VEC__)

namespace {

constexpr uint32_t kWarpSize = SC_WARP_SIZE;
constexpr uint32_t kWarpCount = SC_WARP_COUNT;
constexpr uint32_t kThreadCount = kWarpCount * kWarpSize;
constexpr int kSingleCacheLine = 0;

__simt_callee__ __aicore__ inline uint64_t SimtAtomicLoad(__gm__ volatile int64_t *a) {
    return asc_atomic_add((__gm__ uint64_t *)a, 0);
}

__simt_callee__ __aicore__ inline uint64_t SimtCAS(__gm__ volatile int64_t *a, uint64_t exp, uint64_t des) {
    return asc_atomic_cas((__gm__ uint64_t *)a, exp, des);
}

__simt_callee__ __aicore__ inline uint64_t SimtFetchAdd(__gm__ volatile int64_t *a, uint64_t v) {
    return asc_atomic_add((__gm__ uint64_t *)a, v);
}

__simt_callee__ __aicore__ inline uint32_t SimtCtz64(uint64_t v) {
    uint32_t r = 0;
    while (v && !(v & 1)) { r++; v >>= 1; }
    return r;
}

__simt_callee__ __aicore__ inline void QLock(__gm__ GmQueue *q) {
    while (SimtCAS(&q->lock, 0, 1) != 0) {}
}
__simt_callee__ __aicore__ inline void QUnlock(__gm__ GmQueue *q) {
    (void)SimtCAS(&q->lock, 1, 0);
}

__simt_callee__ __aicore__ inline bool QEnqueue(__gm__ GmQueue *q, uint32_t tid) {
    QLock(q);
    if (q->cnt >= SC_RING_SIZE) { QUnlock(q); return false; }
    q->tasks[q->tail % SC_RING_SIZE] = tid;
    q->tail++; q->cnt++;
    QUnlock(q);
    return true;
}

__simt_callee__ __aicore__ inline bool QDequeue(__gm__ GmQueue *q, uint32_t *out) {
    QLock(q);
    if (q->cnt == 0) { QUnlock(q); return false; }
    *out = q->tasks[q->head % SC_RING_SIZE];
    q->head++; q->cnt--;
    QUnlock(q);
    return true;
}

/* ── desc warp: set state=CREATING + successor cnt=0 ── */
__simt_callee__ __aicore__ inline void RunDescWarp(
    __gm__ ScState *st, uint32_t warp, uint32_t n_desc_warps
) {
    for (uint32_t i = warp; i < st->total_task_cnt; i += n_desc_warps) {
        st->state_buf[i].state = SC_TS_CREATING;
        st->state_buf[i].succ_cnt = 0;
        st->successor_buf[i].cnt = 0;
        st->pred_cnt[i] = 0;
    }
}

/* ── cutter warp: build successor chains + resolve deps ── */
__simt_callee__ __aicore__ inline void RunCutterWarp(__gm__ ScState *st) {
    uint32_t total = st->total_task_cnt;

    /* phase 1: commit all tasks, build successor lists */
    for (uint32_t i = 0; i < total; i++) {
        if (st->predecessors[i].cnt == 0) {
            uint32_t type = st->basic_buf[i].type;
            if (type >= 2) type = SC_TT_VECTOR;
            QEnqueue(&st->ready_queue[type], i);
            st->pred_cnt[i] = 0;
        } else {
            uint32_t pending = 0;
            for (uint32_t j = 0; j < st->predecessors[i].cnt; j++) {
                uint32_t pred = st->pred_ring[st->predecessors[i].exp_offset + j];
                if (st->state_buf[pred].state != SC_TS_COMPLETED) {
                    uint32_t sidx = st->successor_buf[pred].cnt;
                    if (sidx < SC_CON_NODE_CNT)
                        st->successor_buf[pred].node[sidx] = i;
                    st->successor_buf[pred].cnt++;
                    st->state_buf[pred].succ_cnt++;
                    pending++;
                }
            }
            st->pred_cnt[i] = pending;
            if (pending == 0) {
                uint32_t type = st->basic_buf[i].type;
                if (type >= 2) type = SC_TT_VECTOR;
                QEnqueue(&st->ready_queue[type], i);
            }
        }
    }
    st->report_cutter_ready = total;

    /* phase 2: process completed queue + resolve deps */
    while (SimtAtomicLoad(&st->completed_cnt.value) < total) {
        uint32_t tid;
        uint32_t batch = 0;
        while (QDequeue(&st->completed_queue, &tid) && batch < 64) {
            st->state_buf[tid].state = SC_TS_COMPLETED;
            SimtFetchAdd(&st->completed_cnt.value, 1);

            uint32_t sc = st->state_buf[tid].succ_cnt;
            for (uint32_t k = 0; k < sc; k++) {
                uint32_t succ = st->successor_buf[tid].node[k];
                uint64_t prev = SimtFetchAdd((__gm__ volatile int64_t *)&st->pred_cnt[succ], (uint64_t)-1);
                if (prev == 1) {
                    uint32_t stype = st->basic_buf[succ].type;
                    if (stype >= 2) stype = SC_TT_VECTOR;
                    QEnqueue(&st->ready_queue[stype], succ);
                }
            }
            batch++;
        }
    }
}

/* ── dispatch warp: ready_queue → exe_slots (CAS), msg_bitmap → completed_queue ── */
__simt_callee__ __aicore__ inline void RunDispatchWarp(__gm__ ScState *st) {
    uint32_t total = st->total_task_cnt;
    uint32_t sent = 0;

    /* init free_bitmap: all cores free */
    for (uint32_t t = 0; t < 2; t++)
        for (uint32_t s = 0; s < 2; s++)
            st->free_bitmap[t][s] = 0xFFFFFFFFFFFFFFFFULL;

    while (SimtAtomicLoad(&st->completed_cnt.value) < total) {
        /* 1. read completions: msg_bitmap → completed_queue */
        for (uint32_t t = 0; t < 2; t++) {
            for (uint32_t s = 0; s < 2; s++) {
                uint64_t msg = SimtAtomicLoad((__gm__ volatile int64_t *)&st->msg_bitmap[t][s]);
                if (msg == 0) continue;
                /* clear */
                SimtCAS((__gm__ volatile int64_t *)&st->msg_bitmap[t][s], msg, 0);
                while (msg) {
                    uint32_t core = SimtCtz64(msg);
                    uint64_t mask = 1ULL << core;
                    uint32_t slot_idx = core * SC_EXE_OSTD + s;
                    uint32_t tid = st->exe_slots[slot_idx].task_id;
                    QEnqueue(&st->completed_queue, tid);
                    st->free_bitmap[t][s] |= mask;
                    msg &= ~mask;
                }
            }
        }

        /* 2. dispatch: ready_queue → exe_slots */
        for (uint32_t t = 0; t < 2; t++) {
            uint32_t tid;
            while (QDequeue(&st->ready_queue[t], &tid)) {
                /* find free core */
                uint64_t free_bits = st->free_bitmap[t][0] & st->free_bitmap[t][1];
                if (free_bits == 0) {
                    /* no free core, put back */
                    QEnqueue(&st->ready_queue[t], tid);
                    break;
                }
                uint32_t core = SimtCtz64(free_bits);
                uint64_t mask = 1ULL << core;
                uint32_t slot = (st->free_bitmap[t][0] & mask) ? 0 : 1;
                uint32_t slot_idx = core * SC_EXE_OSTD + slot;

                /* write task_id + DCCI + CAS CLAIM */
                st->exe_slots[slot_idx].task_id = tid;
                asc_threadfence();
                asc_atomic_cas((__gm__ uint64_t *)&st->exe_slots[slot_idx].state, SC_SLOT_IDLE, SC_SLOT_CLAIMED);

                st->free_bitmap[t][slot] &= ~mask;
                sent++;
            }
        }
    }
    st->report_dispatch_sent = sent;
}

} /* namespace */

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void
SimtMainVF(
    __gm__ uint64_t *state_words,
    uint64_t warp_count
) {
    __gm__ ScState *st = (__gm__ ScState *)state_words;
    const uint32_t thread = (uint32_t)threadIdx.x;
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;

    if (warp >= (uint32_t)warp_count) return;

    uint32_t n_desc = warp_count / 2;

    if (warp < n_desc) {
        /* desc warps */
        if (lane == 0)
            RunDescWarp(st, warp, n_desc);
        /* barrier: wait for all desc warps */
        asc_atomic_add((__gm__ uint64_t *)&st->desc_done.value, 1);
        while (SimtAtomicLoad(&st->desc_done.value) < n_desc) {}
    } else if (warp == n_desc) {
        /* cutter warp */
        if (lane == 0)
            RunCutterWarp(st);
    } else {
        /* dispatch warp */
        if (lane == 0)
            RunDispatchWarp(st);
    }

    /* all warps: signal all_done */
    if (lane == 0) {
        if (SimtAtomicLoad(&st->completed_cnt.value) >= st->total_task_cnt) {
            SimtCAS(&st->all_done.value, 0, 1);
        }
    }
}

/* SIMD anchor for MIX classification — replaced by RunVectorAnchor with PTO ops */

/* SIMD anchor: forces MIX classification (SIMD+SIMT) */
static __aicore__ __attribute__((noinline, used)) void RunVectorAnchor(
    __gm__ float *input_a, __gm__ float *output, uint32_t repeats
) {
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, 128, 128>,
        pto::Stride<128 * 128, 128 * 128, 128 * 128, 128, 1>>;
    using VecTile = Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1>;
    GlobalData ga(input_a), gc(output);
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
simt_case_aiv(__gm__ ScState *st) {
    dcci((__gm__ void *)&st->phase, kSingleCacheLine);
    dsb(DSB_ALL);

    const uint32_t block = (uint32_t)get_block_idx();
    const uint32_t subdim = (uint32_t)get_subblockdim();
    const uint32_t sub = (uint32_t)get_subblockid();
    const uint32_t aiv_id = block * subdim + sub;

    /* AIV0 = scheduler core */
    if (aiv_id == 0) {
        /* PTO anchor: inline TLOAD+TADD+TSTORE for MIX + .rodata */
        __gm__ float *ws = reinterpret_cast<__gm__ float *>(st->workspace_base);
        __builtin_cce_st_dev(0xDEADBEEFULL, (__gm__ uint64_t *)(ws + 2*128*128), 0);
        dsb(DSB_ALL);
        {
            using GlobalData = GlobalTensor<float, Shape<1,1,1,128,128>,
                pto::Stride<128*128,128*128,128*128,128,1>>;
            using VecTile = Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1>;
            GlobalData ga(ws), gc(ws + 2*128*128);
            VecTile a, b, c;
            TASSIGN(a, 0x0); TASSIGN(b, 0x10000); TASSIGN(c, 0x20000);
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
        dcci((__gm__ void *)(ws + 2*128*128), kSingleCacheLine);
        dsb(DSB_ALL);
        {
            volatile __gm__ uint64_t *out = (volatile __gm__ uint64_t *)(ws + 2*128*128);
            uint64_t cs = *out;
            __builtin_cce_st_dev(cs, (__gm__ uint64_t *)&st->report_magic, 0);
        }

        st->report_start_tick = (uint64_t)get_sys_cnt();

        cce::async_invoke<SimtMainVF>(
            cce::dim3{kThreadCount, 1U, 1U},
            (__gm__ uint64_t *)st,
            (uint64_t)kWarpCount
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        st->report_end_tick = (uint64_t)get_sys_cnt();
        (void)atomicCAS((__gm__ int64_t *)&st->all_done.value, (int64_t)0, (int64_t)1);
        return;
    }

    /* AIV1..63 = vector executor */
    const uint32_t core_id = aiv_id - 1;   /* 0..62 */
    const uint32_t local_core = core_id;

    while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->all_done.value, (int64_t)0) == 0) {
        for (uint32_t slot = 0; slot < SC_EXE_OSTD; slot++) {
            uint32_t idx = local_core * SC_EXE_OSTD + slot;
            __gm__ GmExeSlot *s = &st->exe_slots[idx];
            uint64_t state = (uint64_t)atomicAdd((__gm__ int64_t *)&s->state, (int64_t)0);
            if (state == SC_SLOT_CLAIMED) {
                if ((uint64_t)atomicCAS((__gm__ int64_t *)&s->state, (int64_t)SC_SLOT_CLAIMED, (int64_t)SC_SLOT_DONE) == SC_SLOT_CLAIMED) {
                    uint64_t mask = 1ULL << local_core;
                    atomicAdd((__gm__ uint64_t *)&st->msg_bitmap[SC_TT_VECTOR][slot], mask);
                }
            }
        }
    }
}

#else /* AIC build */

constexpr int kSingleCacheLine = 0;

static __aicore__ __attribute__((noinline, used)) void RunCubeAnchor(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, 128, 128>,
        pto::Stride<128 * 128, 128 * 128, 128 * 128, 128, 1>>;
    using TileMatA = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, 128, 128, 128, 128>;
    using RightTile = TileRight<float, 128, 128, 128, 128>;
    using AccTile = TileAcc<float, 128, 128, 128, 128>;
    GlobalData ga(input_a), gb(input_b), gc(output);
    TileMatA ma; TileMatB mb;
    LeftTile la; RightTile lb; AccTile oc;
    TASSIGN(ma, 0x0); TASSIGN(mb, 0x20000);
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

namespace {

__aicore__ inline uint64_t ScaLoad(__gm__ volatile int64_t *a) {
    return (uint64_t)atomicAdd((__gm__ int64_t *)a, (int64_t)0);
}
__aicore__ inline uint64_t ScaCAS(__gm__ volatile int64_t *a, uint64_t e, uint64_t d) {
    return (uint64_t)atomicCAS((__gm__ int64_t *)a, (int64_t)e, (int64_t)d);
}
__aicore__ inline uint64_t LoadDev64(__gm__ const uint64_t *a) {
    return (uint64_t)__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(a), 0);
}
__aicore__ inline void StoreDev64(__gm__ uint64_t *a, uint64_t v) {
    __builtin_cce_st_dev(v, a, 0);
}

}

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aic(__gm__ ScState *st) {
    dcci((__gm__ void *)&st->phase, kSingleCacheLine);
    dsb(DSB_ALL);

    /* PTO anchor: inline TLOAD+TMATMUL+TSTORE for .rodata + TLV 0x10/0x11 */
    __gm__ float *ws = reinterpret_cast<__gm__ float *>(st->workspace_base);
    StoreDev64((__gm__ uint64_t *)(ws + 2*128*128), 0xDEADBEEFULL);
    dsb(DSB_ALL);
    {
        using GlobalData = GlobalTensor<float, Shape<1,1,1,128,128>,
            pto::Stride<128*128,128*128,128*128,128,1>>;
        using TileMatA = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
        using TileMatB = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
        using LeftTile = TileLeft<float, 128, 128, 128, 128>;
        using RightTile = TileRight<float, 128, 128, 128, 128>;
        using AccTile = TileAcc<float, 128, 128, 128, 128>;
        GlobalData ga(ws), gb(ws + 128*128), gc(ws + 2*128*128);
        TileMatA ma; TileMatB mb;
        LeftTile la; RightTile lb; AccTile oc;
        TASSIGN(ma, 0x0); TASSIGN(mb, 0x20000);
        TASSIGN(la, 0x0); TASSIGN(lb, 0x0); TASSIGN(oc, 0x0);
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
    dcci((__gm__ void *)(ws + 2*128*128), kSingleCacheLine);
    dsb(DSB_ALL);
    {
        volatile __gm__ uint64_t *out = (volatile __gm__ uint64_t *)(ws + 2*128*128);
        uint64_t cs = *out;
        StoreDev64((__gm__ uint64_t *)&st->report_magic, cs);
    }

    const uint32_t owner = (uint32_t)get_block_idx();
    const uint32_t local_core = owner;

    while (ScaLoad(&st->all_done.value) == 0) {
        for (uint32_t slot = 0; slot < SC_EXE_OSTD; slot++) {
            uint32_t idx = local_core * SC_EXE_OSTD + slot;
            __gm__ GmExeSlot *s = &st->exe_slots[idx];
            uint64_t state = ScaLoad((__gm__ volatile int64_t *)&s->state);
            if (state == SC_SLOT_CLAIMED) {
                if (ScaCAS((__gm__ volatile int64_t *)&s->state, SC_SLOT_CLAIMED, SC_SLOT_DONE) == SC_SLOT_CLAIMED) {
                    uint64_t mask = 1ULL << local_core;
                    atomicAdd((__gm__ uint64_t *)&st->msg_bitmap[SC_TT_CUBE][slot], mask);
                }
            }
        }
    }
}

#endif
