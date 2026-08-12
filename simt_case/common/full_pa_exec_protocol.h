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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_EXEC_PROTOCOL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_EXEC_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

namespace pa_scheduler::simt_cross_core::g0 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_G0_INLINE __aicore__ __attribute__((always_inline)) inline
#define SIMT_CROSS_CORE_G0_RUNTIME_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_G0_INLINE constexpr
#define SIMT_CROSS_CORE_G0_RUNTIME_INLINE inline
#endif

constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kTensorDescBytes = 128U;
constexpr uint32_t kTensorDescWords = kTensorDescBytes / sizeof(uint64_t);
constexpr uint32_t kMaxTensors = 32U;
constexpr uint32_t kMaxScalars = 16U;
constexpr uint32_t kMaxFanin = 16U;
constexpr uint32_t kPayloadHeaderWords = kCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kMaxPayloadBytes =
    kCacheLineBytes + kMaxTensors * kTensorDescBytes + kMaxScalars * sizeof(uint64_t) +
    kMaxFanin * sizeof(int32_t);
constexpr uint32_t kMaxPayloadLines = (kMaxPayloadBytes + kCacheLineBytes - 1U) / kCacheLineBytes;
constexpr uint32_t kMaxPayloadWords = kMaxPayloadLines * kPayloadHeaderWords;
constexpr uint32_t kInvalidFunctionId = UINT32_MAX;
constexpr uint32_t kMaxOwner = 254U;
constexpr uint32_t kUnboundOwner = 255U;
constexpr uint32_t kTokensPerOwner = 4U;
constexpr uint32_t kDispatchArgCount = 50U;
constexpr uint32_t kLocalContextBytes = 48U;
constexpr uint32_t kGlobalContextBytes = 4U;
constexpr uint32_t kDispatchBindingBytes = 512U;
constexpr uint32_t kDispatchLocalContextIndex = 48U;
constexpr uint32_t kDispatchGlobalContextIndex = 49U;
constexpr uint32_t kDrainGroupCount = 16U;
constexpr uint32_t kDrainArrivalBits = 8U;
constexpr uint64_t kDrainArrivalMask = (uint64_t{1} << kDrainArrivalBits) - 1U;

enum class ExecPhase : uint8_t {
    Empty = 0,
    Building = 1,
    Built = 2,
    Claimed = 3,
    Done = 4,
};

enum class ExecEngineClass : uint8_t {
    None = 0,
    Aic = 1,
    Aiv = 2,
    Joint = 3,
};

enum class ExecTokenPhase : uint32_t {
    Idle = 0,
    Binding = 1,
    WaitingFanin = 2,
    EngineInflight = 3,
    Completing = 4,
    VendPublished = 5,
    CompletionPublished = 6,
    Faulted = 7,
    WaitingBuilt = 8,
};

enum class ExecFatalReason : uint8_t {
    None = 0,
    InvalidBuildInput = 1,
    BuildPackFailed = 2,
    InvalidBuiltControl = 3,
    ClaimedPayloadInvalid = 4,
    ControlPublishConflict = 5,
    InvalidTokenPayload = 6,
    CompletionPublishFailed = 7,
    CompletionStateConflict = 8,
    Timeout = 9,
    HeapReservationFailed = 10,
    InsertProtocolFailed = 11,
    DrainMismatch = 12,
};

constexpr uint64_t kStatePhaseShift = 0U;
constexpr uint64_t kStatePhaseMask = 0x7ULL;
constexpr uint64_t kStateBuildOwnerShift = 3U;
constexpr uint64_t kStateBuildOwnerMask = 0xFFULL;
constexpr uint64_t kStateExecuteOwnerShift = 11U;
constexpr uint64_t kStateExecuteOwnerMask = 0xFFULL;
constexpr uint64_t kStateEngineShift = 19U;
constexpr uint64_t kStateEngineMask = 0x7ULL;
constexpr uint64_t kStatePayloadLinesShift = 22U;
constexpr uint64_t kStatePayloadLinesMask = 0x7FULL;
constexpr uint64_t kStateTaskIdShift = 29U;
constexpr uint64_t kStateTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kStateKnownMask =
    (kStatePhaseMask << kStatePhaseShift) | (kStateBuildOwnerMask << kStateBuildOwnerShift) |
    (kStateExecuteOwnerMask << kStateExecuteOwnerShift) | (kStateEngineMask << kStateEngineShift) |
    (kStatePayloadLinesMask << kStatePayloadLinesShift) | (kStateTaskIdMask << kStateTaskIdShift);

constexpr uint64_t kFatalReasonShift = 0U;
constexpr uint64_t kFatalReasonMask = 0xFFULL;
constexpr uint64_t kFatalOwnerShift = 8U;
constexpr uint64_t kFatalOwnerMask = 0xFFULL;
constexpr uint64_t kFatalTaskIdShift = 16U;
constexpr uint64_t kFatalTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kFatalKnownMask =
    (kFatalReasonMask << kFatalReasonShift) | (kFatalOwnerMask << kFatalOwnerShift) |
    (kFatalTaskIdMask << kFatalTaskIdShift);

struct DecodedExecState {
    ExecPhase phase;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

struct DecodedExecFatal {
    ExecFatalReason reason;
    uint32_t reporter_owner;
    uint32_t task_id;
    bool valid;
};

struct ExecPayloadLayout {
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t tensor_word_offset;
    uint32_t scalar_word_offset;
    uint32_t fanin_word_offset;
    uint32_t written_words;
    uint32_t tensor_reference_mask;
    uint32_t inline_tensor_count;
};

struct ExecPayloadHeader {
    uint32_t task_id;
    uint64_t function_address;
    uint64_t completion_vend;
    uint32_t function_id;
    uint32_t payload_bytes;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t multicore_group_id;
    uint16_t multicore_rank;
    uint16_t multicore_size;
    uint32_t tensor_reference_mask;
};

struct alignas(kCacheLineBytes) AtomicLine {
    volatile int64_t value;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) SharedExecControl {
    volatile int64_t state;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) ExecPayloadStorage {
    volatile uint64_t words[kMaxPayloadWords];
};

struct alignas(kCacheLineBytes) SharedExecCell {
    SharedExecControl control;
    ExecPayloadStorage payload;
};

struct alignas(kCacheLineBytes) ExecutionTokenControl {
    ExecTokenPhase phase;
    uint32_t task_id;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t payload_bytes;
    uint32_t fanin_ready_prefix;
    uint64_t payload_address;
    uint64_t completion_vend;
    uint64_t function_and_reference;
    uint64_t shape_and_scalar_offset;
};

struct alignas(kCacheLineBytes) ExecutionDispatchBinding {
    uint64_t args[kDispatchArgCount];
    uint8_t local_context[kLocalContextBytes];
    uint8_t global_context[kGlobalContextBytes];
    uint8_t padding[
        kDispatchBindingBytes - kDispatchArgCount * sizeof(uint64_t) - kLocalContextBytes - kGlobalContextBytes
    ];
};

struct alignas(kCacheLineBytes) ExecutionToken {
    ExecutionTokenControl control;
    ExecutionDispatchBinding dispatch;
};

SIMT_CROSS_CORE_G0_INLINE bool ExecOwnerValid(uint32_t owner) { return owner <= kMaxOwner; }

SIMT_CROSS_CORE_G0_INLINE bool ExecEngineValid(ExecEngineClass engine_class) {
    return engine_class == ExecEngineClass::Aic || engine_class == ExecEngineClass::Aiv ||
           engine_class == ExecEngineClass::Joint;
}

SIMT_CROSS_CORE_G0_INLINE uint64_t EncodeExecState(
    ExecPhase phase, uint32_t build_owner, uint32_t execute_owner, ExecEngineClass engine_class,
    uint32_t payload_lines, uint32_t task_id
) {
    return (static_cast<uint64_t>(phase) << kStatePhaseShift) |
           (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
           (static_cast<uint64_t>(execute_owner) << kStateExecuteOwnerShift) |
           (static_cast<uint64_t>(engine_class) << kStateEngineShift) |
           (static_cast<uint64_t>(payload_lines) << kStatePayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
}

SIMT_CROSS_CORE_G0_INLINE DecodedExecState DecodeExecState(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedExecState state{
        static_cast<ExecPhase>((raw >> kStatePhaseShift) & kStatePhaseMask),
        static_cast<uint32_t>((raw >> kStateBuildOwnerShift) & kStateBuildOwnerMask),
        static_cast<uint32_t>((raw >> kStateExecuteOwnerShift) & kStateExecuteOwnerMask),
        static_cast<ExecEngineClass>((raw >> kStateEngineShift) & kStateEngineMask),
        static_cast<uint32_t>((raw >> kStatePayloadLinesShift) & kStatePayloadLinesMask),
        static_cast<uint32_t>((raw >> kStateTaskIdShift) & kStateTaskIdMask),
        false,
    };
    if ((raw & ~kStateKnownMask) != 0U) {
        return state;
    }
    if (state.phase == ExecPhase::Empty) {
        state.valid = state.build_owner == 0U && state.execute_owner == 0U &&
                      state.engine_class == ExecEngineClass::None && state.payload_lines == 0U &&
                      state.task_id == 0U;
    } else if (state.phase == ExecPhase::Building) {
        state.valid = ExecOwnerValid(state.build_owner) && state.execute_owner == kUnboundOwner &&
                      state.engine_class == ExecEngineClass::None && state.payload_lines == 0U;
    } else if (state.phase == ExecPhase::Built) {
        state.valid = ExecOwnerValid(state.build_owner) && state.execute_owner == kUnboundOwner &&
                      ExecEngineValid(state.engine_class) && state.payload_lines >= 1U &&
                      state.payload_lines <= kMaxPayloadLines;
    } else if (state.phase == ExecPhase::Claimed || state.phase == ExecPhase::Done) {
        state.valid = ExecOwnerValid(state.build_owner) && ExecOwnerValid(state.execute_owner) &&
                      ExecEngineValid(state.engine_class) && state.payload_lines >= 1U &&
                      state.payload_lines <= kMaxPayloadLines;
    }
    return state;
}

SIMT_CROSS_CORE_G0_INLINE uint64_t EncodeExecFatal(
    ExecFatalReason reason, uint32_t reporter_owner, uint32_t task_id
) {
    return (static_cast<uint64_t>(reason) << kFatalReasonShift) |
           (static_cast<uint64_t>(reporter_owner) << kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << kFatalTaskIdShift);
}

SIMT_CROSS_CORE_G0_INLINE DecodedExecFatal DecodeExecFatal(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    const ExecFatalReason reason = static_cast<ExecFatalReason>((raw >> kFatalReasonShift) & kFatalReasonMask);
    const uint32_t owner = static_cast<uint32_t>((raw >> kFatalOwnerShift) & kFatalOwnerMask);
    const uint32_t task_id = static_cast<uint32_t>((raw >> kFatalTaskIdShift) & kFatalTaskIdMask);
    return DecodedExecFatal{
        reason,
        owner,
        task_id,
        raw != 0U && (raw & ~kFatalKnownMask) == 0U && reason >= ExecFatalReason::InvalidBuildInput &&
            reason <= ExecFatalReason::DrainMismatch && ExecOwnerValid(owner),
    };
}

SIMT_CROSS_CORE_G0_INLINE uint32_t TensorMaskForCount(uint32_t tensor_count) {
    return tensor_count >= 32U ? UINT32_MAX : ((uint32_t{1} << tensor_count) - 1U);
}

SIMT_CROSS_CORE_G0_INLINE uint32_t PopCount32(uint32_t value) {
    uint32_t count = 0U;
    while (value != 0U) {
        count += value & 1U;
        value >>= 1U;
    }
    return count;
}

SIMT_CROSS_CORE_G0_INLINE uint32_t TensorPayloadWordOffset(uint32_t tensor, uint32_t reference_mask) {
    const uint32_t preceding_mask = tensor == 0U ? 0U : reference_mask & TensorMaskForCount(tensor);
    return kPayloadHeaderWords + tensor * kTensorDescWords -
           PopCount32(preceding_mask) * (kTensorDescWords - 1U);
}

SIMT_CROSS_CORE_G0_INLINE bool ComputeExecPayloadLayout(
    uint32_t tensor_count, uint32_t scalar_count, uint32_t fanin_count, uint32_t tensor_reference_mask,
    ExecPayloadLayout &layout
) {
    if (tensor_count > kMaxTensors || scalar_count > kMaxScalars || fanin_count > kMaxFanin ||
        (tensor_reference_mask & ~TensorMaskForCount(tensor_count)) != 0U) {
        return false;
    }
    const uint32_t reference_count = PopCount32(tensor_reference_mask);
    const uint32_t inline_tensor_count = tensor_count - reference_count;
    layout.tensor_word_offset = kPayloadHeaderWords;
    layout.scalar_word_offset =
        layout.tensor_word_offset + inline_tensor_count * kTensorDescWords + reference_count;
    layout.fanin_word_offset = layout.scalar_word_offset + scalar_count;
    layout.written_words = layout.fanin_word_offset + (fanin_count + 1U) / 2U;
    layout.payload_bytes = kCacheLineBytes + inline_tensor_count * kTensorDescBytes +
                           reference_count * sizeof(uint64_t) + scalar_count * sizeof(uint64_t) +
                           fanin_count * sizeof(int32_t);
    layout.payload_lines = (layout.payload_bytes + kCacheLineBytes - 1U) / kCacheLineBytes;
    layout.tensor_reference_mask = tensor_reference_mask;
    layout.inline_tensor_count = inline_tensor_count;
    return layout.payload_lines >= 1U && layout.payload_lines <= kMaxPayloadLines &&
           layout.written_words <= kMaxPayloadWords;
}

SIMT_CROSS_CORE_G0_INLINE bool ComputeExecPayloadLayout(
    uint32_t tensor_count, uint32_t scalar_count, uint32_t fanin_count, ExecPayloadLayout &layout
) {
    return ComputeExecPayloadLayout(tensor_count, scalar_count, fanin_count, 0U, layout);
}

SIMT_CROSS_CORE_G0_INLINE uint64_t PackHeaderWord3(uint32_t function_id, uint32_t payload_bytes) {
    return static_cast<uint64_t>(function_id) | (static_cast<uint64_t>(payload_bytes) << 32U);
}

SIMT_CROSS_CORE_G0_INLINE uint64_t PackHeaderWord4(
    uint16_t tensor_count, uint16_t scalar_count, uint16_t fanin_count, ExecEngineClass engine_class
) {
    return static_cast<uint64_t>(tensor_count) | (static_cast<uint64_t>(scalar_count) << 16U) |
           (static_cast<uint64_t>(fanin_count) << 32U) |
           (static_cast<uint64_t>(engine_class) << 48U);
}

SIMT_CROSS_CORE_G0_INLINE uint64_t PackExecutionTokenShapeAndScalarOffset(
    uint16_t tensor_count, uint16_t scalar_count, uint16_t fanin_count, uint16_t scalar_word_offset
) {
    return static_cast<uint64_t>(tensor_count) | (static_cast<uint64_t>(scalar_count) << 16U) |
           (static_cast<uint64_t>(fanin_count) << 32U) |
           (static_cast<uint64_t>(scalar_word_offset) << 48U);
}

SIMT_CROSS_CORE_G0_RUNTIME_INLINE ExecPayloadHeader DecodeExecPayloadHeader(const ExecPayloadStorage &payload) {
    const uint64_t word0 = payload.words[0];
    const uint64_t word3 = payload.words[3];
    const uint64_t word4 = payload.words[4];
    const uint64_t word5 = payload.words[5];
    return ExecPayloadHeader{
        static_cast<uint32_t>(word0),
        payload.words[1],
        payload.words[2],
        static_cast<uint32_t>(word3),
        static_cast<uint32_t>(word3 >> 32U),
        static_cast<uint16_t>(word4),
        static_cast<uint16_t>(word4 >> 16U),
        static_cast<uint16_t>(word4 >> 32U),
        static_cast<ExecEngineClass>(static_cast<uint8_t>(word4 >> 48U)),
        static_cast<uint8_t>(word4 >> 56U),
        static_cast<uint32_t>(word5),
        static_cast<uint16_t>(word5 >> 32U),
        static_cast<uint16_t>(word5 >> 48U),
        static_cast<uint32_t>(payload.words[6]),
    };
}

SIMT_CROSS_CORE_G0_INLINE int64_t EncodeDrainContribution(uint64_t completed_tasks) {
    return completed_tasks > (static_cast<uint64_t>(INT64_MAX) >> kDrainArrivalBits)
               ? -1
               : static_cast<int64_t>((completed_tasks << kDrainArrivalBits) | 1U);
}

SIMT_CROSS_CORE_G0_INLINE uint32_t DecodeDrainArrivals(int64_t raw) {
    return raw < 0 ? UINT32_MAX : static_cast<uint32_t>(static_cast<uint64_t>(raw) & kDrainArrivalMask);
}

SIMT_CROSS_CORE_G0_INLINE uint64_t DecodeDrainCompletions(int64_t raw) {
    return raw < 0 ? UINT64_MAX : static_cast<uint64_t>(raw) >> kDrainArrivalBits;
}

static_assert(kMaxPayloadBytes == 4352U && kMaxPayloadLines == 68U, "execution payload capacity changed");
static_assert(sizeof(AtomicLine) == kCacheLineBytes, "atomic line ABI changed");
static_assert(sizeof(SharedExecControl) == kCacheLineBytes, "execution control ABI changed");
static_assert(sizeof(ExecPayloadStorage) == kMaxPayloadBytes, "execution payload storage ABI changed");
static_assert(offsetof(SharedExecCell, payload) == kCacheLineBytes, "payload must follow its control line");
static_assert(sizeof(SharedExecCell) == 4416U, "execution cell must remain exactly 4416 bytes");
static_assert(sizeof(ExecutionTokenControl) == kCacheLineBytes, "execution token control ABI changed");
static_assert(sizeof(ExecutionDispatchBinding) == kDispatchBindingBytes, "dispatch binding ABI changed");
static_assert(sizeof(ExecutionToken) == 576U, "each execution token must remain exactly 576 bytes");
static_assert(offsetof(ExecutionToken, dispatch) == kCacheLineBytes, "token dispatch offset changed");
static_assert(kMaxTensors + kMaxScalars == kDispatchLocalContextIndex &&
                  kDispatchGlobalContextIndex + 1U == kDispatchArgCount,
              "dispatch context indexes no longer match the PA ABI");
static_assert(static_cast<uint32_t>(ExecPhase::Empty) == 0U &&
                  static_cast<uint32_t>(ExecEngineClass::None) == 0U,
              "zeroed control must encode EMPTY");
#if !defined(__CCE_AICORE__)
static_assert(
    PackExecutionTokenShapeAndScalarOffset(7U, 2U, 3U, 120U) ==
        (uint64_t{7U} | (uint64_t{2U} << 16U) | (uint64_t{3U} << 32U) | (uint64_t{120U} << 48U)),
    "execution token shape metadata no longer matches the production ABI"
);
#endif

#undef SIMT_CROSS_CORE_G0_INLINE
#undef SIMT_CROSS_CORE_G0_RUNTIME_INLINE

}  // namespace pa_scheduler::simt_cross_core::g0

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_EXEC_PROTOCOL_H
