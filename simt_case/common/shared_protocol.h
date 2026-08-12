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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_SHARED_PROTOCOL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_SHARED_PROTOCOL_H

#include <stdint.h>

namespace pa_scheduler::simt_cross_core {

constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kMaxPayloadLines = 68U;
constexpr uint32_t kMaxOwner = 254U;
constexpr uint32_t kUnboundOwner = 255U;

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

struct DecodedExecState {
    ExecPhase phase;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

constexpr uint64_t EncodeExecState(
    ExecPhase phase, uint32_t build_owner, uint32_t execute_owner, ExecEngineClass engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    return (static_cast<uint64_t>(phase) << kStatePhaseShift) |
           (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
           (static_cast<uint64_t>(execute_owner) << kStateExecuteOwnerShift) |
           (static_cast<uint64_t>(engine_class) << kStateEngineShift) |
           (static_cast<uint64_t>(payload_lines) << kStatePayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
}

constexpr bool ExecOwnerValid(uint32_t owner) { return owner <= kMaxOwner; }

constexpr bool ExecEngineValid(ExecEngineClass engine_class) {
    return engine_class == ExecEngineClass::Aic || engine_class == ExecEngineClass::Aiv ||
           engine_class == ExecEngineClass::Joint;
}

constexpr DecodedExecState DecodeExecState(uint64_t raw) {
    const auto phase = static_cast<ExecPhase>((raw >> kStatePhaseShift) & kStatePhaseMask);
    const uint32_t build_owner = static_cast<uint32_t>((raw >> kStateBuildOwnerShift) & kStateBuildOwnerMask);
    const uint32_t execute_owner = static_cast<uint32_t>((raw >> kStateExecuteOwnerShift) & kStateExecuteOwnerMask);
    const auto engine_class = static_cast<ExecEngineClass>((raw >> kStateEngineShift) & kStateEngineMask);
    const uint32_t payload_lines = static_cast<uint32_t>((raw >> kStatePayloadLinesShift) & kStatePayloadLinesMask);
    const uint32_t task_id = static_cast<uint32_t>((raw >> kStateTaskIdShift) & kStateTaskIdMask);
    DecodedExecState state{
        phase, build_owner, execute_owner, engine_class, payload_lines, task_id, false,
    };
    if ((raw & ~kStateKnownMask) != 0U) {
        return state;
    }
    switch (state.phase) {
    case ExecPhase::Empty:
        state.valid = state.build_owner == 0U && state.execute_owner == 0U &&
                      state.engine_class == ExecEngineClass::None && state.payload_lines == 0U && state.task_id == 0U;
        break;
    case ExecPhase::Building:
        state.valid = ExecOwnerValid(state.build_owner) && state.execute_owner == kUnboundOwner &&
                      state.engine_class == ExecEngineClass::None && state.payload_lines == 0U;
        break;
    case ExecPhase::Built:
        state.valid = ExecOwnerValid(state.build_owner) && state.execute_owner == kUnboundOwner &&
                      ExecEngineValid(state.engine_class) && state.payload_lines >= 1U &&
                      state.payload_lines <= kMaxPayloadLines;
        break;
    case ExecPhase::Claimed:
    case ExecPhase::Done:
        state.valid = ExecOwnerValid(state.build_owner) && ExecOwnerValid(state.execute_owner) &&
                      ExecEngineValid(state.engine_class) && state.payload_lines >= 1U &&
                      state.payload_lines <= kMaxPayloadLines;
        break;
    default:
        break;
    }
    return state;
}

enum class ExecFatalReason : uint8_t {
    None = 0,
    InvalidBuildInput = 1,
    PublishConflict = 2,
    InvalidBuiltControl = 3,
    InvalidPayload = 4,
    CompletionConflict = 5,
    Timeout = 6,
};

constexpr uint64_t kFatalReasonShift = 0U;
constexpr uint64_t kFatalReasonMask = 0xFFULL;
constexpr uint64_t kFatalOwnerShift = 8U;
constexpr uint64_t kFatalOwnerMask = 0xFFULL;
constexpr uint64_t kFatalTaskIdShift = 16U;
constexpr uint64_t kFatalTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kFatalKnownMask = (kFatalReasonMask << kFatalReasonShift) | (kFatalOwnerMask << kFatalOwnerShift) |
                                     (kFatalTaskIdMask << kFatalTaskIdShift);

struct DecodedExecFatal {
    ExecFatalReason reason;
    uint32_t reporter_owner;
    uint32_t task_id;
    bool valid;
};

constexpr uint64_t EncodeExecFatal(ExecFatalReason reason, uint32_t reporter_owner, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << kFatalReasonShift) |
           (static_cast<uint64_t>(reporter_owner) << kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << kFatalTaskIdShift);
}

constexpr DecodedExecFatal DecodeExecFatal(uint64_t raw) {
    const auto reason = static_cast<ExecFatalReason>((raw >> kFatalReasonShift) & kFatalReasonMask);
    const uint32_t owner = static_cast<uint32_t>((raw >> kFatalOwnerShift) & kFatalOwnerMask);
    const uint32_t task_id = static_cast<uint32_t>((raw >> kFatalTaskIdShift) & kFatalTaskIdMask);
    return DecodedExecFatal{
        reason,
        owner,
        task_id,
        raw != 0U && (raw & ~kFatalKnownMask) == 0U && reason >= ExecFatalReason::InvalidBuildInput &&
            reason <= ExecFatalReason::Timeout && owner <= kMaxOwner,
    };
}

static_assert(
    EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U) == 0U,
    "host-zeroed execution state must encode EMPTY"
);
static_assert(kStateTaskIdShift + 32U <= 64U, "execution state exceeds 64 bits");

}  // namespace pa_scheduler::simt_cross_core

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_SHARED_PROTOCOL_H
