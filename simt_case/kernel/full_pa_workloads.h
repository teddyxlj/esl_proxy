/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_WORKLOADS_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_WORKLOADS_H

namespace pa_scheduler::simt_cross_core::g0::device {

constexpr int kG0WorkloadTile = 128;

#if defined(__DAV_VEC__)

template <bool Multiply>
__aicore__ __attribute__((always_inline)) inline void RunG0VectorWorkload(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kG0WorkloadTile, kG0WorkloadTile>,
        pto::Stride<1, 1, 1, kG0WorkloadTile, 1>>;
    using TileData =
        Tile<TileType::Vec, float, kG0WorkloadTile, kG0WorkloadTile, BLayout::RowMajor, -1, -1>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileData input_a_tile(kG0WorkloadTile, kG0WorkloadTile);
    TileData input_b_tile(kG0WorkloadTile, kG0WorkloadTile);
    TileData output_tile(kG0WorkloadTile, kG0WorkloadTile);
    TASSIGN(input_a_tile, 0x0);
    TASSIGN(input_b_tile, 0x10000);
    TASSIGN(output_tile, 0x20000);

    for (uint32_t iteration = 0U; iteration < repeats; ++iteration) {
        TLOAD(input_a_tile, input_a_global);
        TLOAD(input_b_tile, input_b_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (Multiply) {
            TMUL(output_tile, input_a_tile, input_b_tile);
        } else {
            TADD(output_tile, input_a_tile, input_b_tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(output_global, output_tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}

static __aicore__ __attribute__((noinline, used)) void RunG0VectorAdd(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    RunG0VectorWorkload<false>(input_a, input_b, output, repeats);
}

static __aicore__ __attribute__((noinline, used)) void RunG0VectorMultiply(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    RunG0VectorWorkload<true>(input_a, input_b, output, repeats);
}

#else

static __aicore__ __attribute__((noinline, used)) void RunG0CubeMatmul(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    constexpr int kBlockAlign = C0_SIZE_BYTE / sizeof(float);
    static_assert(kG0WorkloadTile % 16 == 0, "G0 Cube M must be 16-aligned");
    static_assert(kG0WorkloadTile % kBlockAlign == 0, "G0 Cube K/N must satisfy C0 alignment");

    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kG0WorkloadTile, kG0WorkloadTile>,
        pto::Stride<
            kG0WorkloadTile * kG0WorkloadTile, kG0WorkloadTile * kG0WorkloadTile,
            kG0WorkloadTile * kG0WorkloadTile, kG0WorkloadTile, 1>>;
    using TileMatA = Tile<
        TileType::Mat, float, kG0WorkloadTile, kG0WorkloadTile, BLayout::ColMajor, kG0WorkloadTile,
        kG0WorkloadTile, SLayout::RowMajor, 512>;
    using TileMatB = Tile<
        TileType::Mat, float, kG0WorkloadTile, kG0WorkloadTile, BLayout::ColMajor, kG0WorkloadTile,
        kG0WorkloadTile, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<
        float, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile>;
    using RightTile = TileRight<
        float, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile>;
    using AccTile =
        TileAcc<float, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile, kG0WorkloadTile>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileMatA input_a_mat;
    TileMatB input_b_mat;
    LeftTile input_a_l0;
    RightTile input_b_l0;
    AccTile output_l0;
    TASSIGN(input_a_mat, 0x0);
    TASSIGN(input_b_mat, 0x20000);
    TASSIGN(input_a_l0, 0x0);
    TASSIGN(input_b_l0, 0x0);
    TASSIGN(output_l0, 0x0);

    for (uint32_t iteration = 0U; iteration < repeats; ++iteration) {
        TLOAD(input_a_mat, input_a_global);
        TLOAD(input_b_mat, input_b_global);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(input_a_l0, input_a_mat);
        TMOV(input_b_l0, input_b_mat);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(output_l0, input_a_l0, input_b_l0);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE(output_global, output_l0);
        set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    }
}

#endif

}  // namespace pa_scheduler::simt_cross_core::g0::device

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_WORKLOADS_H
