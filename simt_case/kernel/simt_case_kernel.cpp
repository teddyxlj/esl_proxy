#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>
#include "cce_aicore_intrinsics.h"

#include "../common/simt_case_protocol.h"
#include "../common/g0_full_pa.h"
#include "full_pa_workloads.h"

using namespace pa_scheduler::simt_cross_core::g0;
using namespace pa_scheduler::simt_cross_core::g0::device;

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

__simt_callee__ __aicore__ inline uint64_t SimtFetchAdd(__gm__ volatile int64_t *a, uint64_t v) {
    return asc_atomic_add((__gm__ uint64_t *)a, v);
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

    __gm__ ScState *st = (__gm__ ScState *)state_words;
    for (uint32_t i = warp; i < (uint32_t)total_task_cnt; i += (uint32_t)warp_count) {
        st->state_buf[i].state = SC_TS_CREATING;
        st->state_buf[i].succ_cnt = 0;
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

        /* PTO anchor via always_inline wrapper */
        {
            __gm__ float *ws = (__gm__ float *)st->workspace_base;
            __gm__ float *input_a = ws;
            __gm__ float *output = ws + 2 * kG0WorkloadTile * kG0WorkloadTile;
            uint64_t poison = 0xDEADBEEFULL;
            __builtin_cce_st_dev(poison, (__gm__ uint64_t *)output, 0);
            dsb(DSB_ALL);
            RunG0VectorAdd(input_a, input_a, output, 1u);
            uint64_t cs = (uint64_t)__builtin_cce_ld_dev((__gm__ uint64_t *)output, 0);
            if (cs == poison) {
                __builtin_cce_st_dev(0ULL, (__gm__ uint64_t *)&st->report_desc_writes, 0);
            } else {
                __builtin_cce_st_dev(cs, (__gm__ uint64_t *)&st->report_desc_writes, 0);
            }
        }

        st->report_start_tick = (uint64_t)get_sys_cnt();

        cce::async_invoke<SimtDescVF>(
            cce::dim3{kThreadCount, 1U, 1U},
            (__gm__ uint64_t *)st,
            (uint64_t)st->total_task_cnt,
            (uint64_t)kWarpCount
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        st->report_end_tick = (uint64_t)get_sys_cnt();
        (void)atomicCAS((__gm__ int64_t *)&st->desc_done.value, (int64_t)0, (int64_t)1);
    }

    while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->all_done.value, (int64_t)0) == 0) {}
}

#else /* AIC build */

__aicore__ __attribute__((always_inline)) inline void
RunAnchorAic(__gm__ ScState *st) {
    __gm__ float *ws = (__gm__ float *)st->workspace_base;
    __gm__ float *input_a = ws;
    __gm__ float *input_b = ws + kG0WorkloadTile * kG0WorkloadTile;
    __gm__ float *output = ws + 2 * kG0WorkloadTile * kG0WorkloadTile;
    uint64_t poison = 0xDEADBEEFULL;
    __builtin_cce_st_dev(poison, (__gm__ uint64_t *)output, 0);
    dsb(DSB_ALL);
    RunG0CubeMatmul(input_a, input_b, output, 1u);
    uint64_t cs = (uint64_t)__builtin_cce_ld_dev((__gm__ uint64_t *)output, 0);
    if (cs == poison) {
        __builtin_cce_st_dev(0ULL, (__gm__ uint64_t *)&st->report_desc_writes, 0);
    } else {
        __builtin_cce_st_dev(cs, (__gm__ uint64_t *)&st->report_desc_writes, 0);
    }
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aic(__gm__ ScState *st) {
    dcci((__gm__ void *)&st->phase, 0);
    dsb(DSB_ALL);

    RunAnchorAic(st);

    while ((uint64_t)atomicAdd((__gm__ int64_t *)&st->all_done.value, (int64_t)0) == 0) {}
}

#endif
