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

/* SIMD anchor for MIX classification */
__simd_vf__ __aicore__ void simt_case_simd_anchor(__ubuf__ uint32_t *s) {
    s[0] = s[0] + 1U;
}

#if defined(__DAV_VEC__)

namespace {

constexpr uint32_t kWarpSize = SC_WARP_SIZE;
constexpr uint32_t kWarpCount = SC_WARP_COUNT;
constexpr uint32_t kThreadCount = kWarpCount * kWarpSize;

__simt_callee__ __aicore__ inline uint64_t SimtAtomicLoad(__gm__ volatile int64_t *a) {
    return asc_atomic_add((__gm__ uint64_t *)a, 0);
}

/* SIMT desc VF: warp-interleaved set state=CREATING (alloc already done by AICPU) */
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

    __gm__ ScState *st = (__gm__ ScState *)state_words;

    for (uint32_t i = warp; i < (uint32_t)total_task_cnt; i += (uint32_t)warp_count) {
        /* verify task is already filled by AICPU alloc */
        /* set state=CREATING (already set by alloc, but re-affirm) */
        st->state_buf[i].state = SC_TS_CREATING;
        st->successor_buf[i].cnt = 0;
        st->pred_cnt[i] = 0;
        if (lane == 0)
            asc_atomic_add((__gm__ uint64_t *)&st->report_desc_writes, 1);
    }
}

} /* namespace */

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aiv(__gm__ ScState *st) {
    dcci((__gm__ void *)&st->phase, 0);
    dsb(DSB_ALL);

    const uint32_t block = (uint32_t)get_block_idx();
    const uint32_t subdim = (uint32_t)get_subblockdim();
    const uint32_t sub = (uint32_t)get_subblockid();
    const uint32_t aiv_id = block * subdim + sub;

    if (aiv_id == 0) {
        /* SIMD anchor for MIX classification */
        __ubuf__ uint32_t scratch[1]; scratch[0] = 0;
        simt_case_simd_anchor(scratch);

        /* PTO anchor: TLOAD+TSTORE from state to generate .rodata + TLV 0x10 */
        {
            using GlobalData = GlobalTensor<float, Shape<1,1,1,128,128>,
                pto::Stride<128*128,128*128,128*128,128,1>>;
            using VecTile = Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1>;
            GlobalData ga((__gm__ float *)&st->basic_buf[0]);
            GlobalData gc((__gm__ float *)&st->basic_buf[1]);
            VecTile a, c;
            TASSIGN(a, 0x0); TASSIGN(c, 0x0);
            TLOAD(a, ga);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TSTORE(gc, a);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        }

        /* wait for alloc_done (AICPU sets phase=1) */
        while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->phase.value, (int64_t)0) < 1) {}

        /* run SIMT desc VF */
        cce::async_invoke<SimtDescVF>(
            cce::dim3{kThreadCount, 1U, 1U},
            (__gm__ uint64_t *)st,
            (uint64_t)st->total_task_cnt,
            (uint64_t)kWarpCount
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        /* signal desc_done (AICPU sched waits for this) */
        (void)atomicCAS((__gm__ int64_t *)&st->phase.value, (int64_t)1, (int64_t)2);
    }

    /* AIV executor: poll exe_slots for CLAIMED → DONE (immediate complete) */
    const uint32_t core_id = aiv_id;
    if (aiv_id > 0) {
        const uint32_t local_core = core_id - 1;
        while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->phase.value, (int64_t)0) < 3) {
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
    } else {
        /* AIV0: wait for all_done */
        while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->phase.value, (int64_t)0) < 3) {}
    }
}

#else /* AIC build */

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aic(__gm__ ScState *st) {
    dcci((__gm__ void *)&st->phase, 0);
    dsb(DSB_ALL);

    /* PTO anchor: TLOAD+TMATMUL+TSTORE for .rodata + TLV 0x10 */
    {
        using GlobalData = GlobalTensor<float, Shape<1,1,1,128,128>,
            pto::Stride<128*128,128*128,128*128,128,1>>;
        using TileMatA = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
        using TileMatB = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
        using LeftTile = TileLeft<float, 128, 128, 128, 128>;
        using RightTile = TileRight<float, 128, 128, 128, 128>;
        using AccTile = TileAcc<float, 128, 128, 128, 128>;
        GlobalData ga((__gm__ float *)&st->basic_buf[0]);
        GlobalData gb((__gm__ float *)&st->basic_buf[1]);
        GlobalData gc((__gm__ float *)&st->basic_buf[2]);
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

    const uint32_t owner = (uint32_t)get_block_idx();
    const uint32_t local_core = owner;

    /* AIC executor: poll exe_slots for CLAIMED → DONE */
    while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->phase.value, (int64_t)0) < 3) {
        for (uint32_t slot = 0; slot < SC_EXE_OSTD; slot++) {
            uint32_t idx = local_core * SC_EXE_OSTD + slot;
            __gm__ GmExeSlot *s = &st->exe_slots[idx];
            uint64_t state = (uint64_t)atomicAdd((__gm__ int64_t *)&s->state, (int64_t)0);
            if (state == SC_SLOT_CLAIMED) {
                if ((uint64_t)atomicCAS((__gm__ int64_t *)&s->state, (int64_t)SC_SLOT_CLAIMED, (int64_t)SC_SLOT_DONE) == SC_SLOT_CLAIMED) {
                    uint64_t mask = 1ULL << local_core;
                    atomicAdd((__gm__ uint64_t *)&st->msg_bitmap[SC_TT_CUBE][slot], mask);
                }
            }
        }
    }
}

#endif
