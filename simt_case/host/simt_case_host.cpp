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

#include "../common/g0_full_pa.h"
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
#include "../common/g0_swimlane.h"
#elif defined(SIMT_CROSS_CORE_U2)
#include "../../ubuf/common/u2_full_pa.h"
#endif

#include "acl/acl.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace g0 = pa_scheduler::simt_cross_core::g0;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
namespace g0_swimlane = pa_scheduler::simt_cross_core::g0_swimlane;
#elif defined(SIMT_CROSS_CORE_U2)
namespace u2 = pa_scheduler::simt_cross_core::u2;
namespace ubuf_staging = pa_scheduler::simt_cross_core::ubuf_staging;
#endif
using pa_scheduler::simt_cross_core::g0::FullPaState;
using pa_scheduler::simt_cross_core::g0::kWorkloadBytes;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
using LaunchState = pa_scheduler::simt_cross_core::g0_swimlane::G0SwimlaneState;
#elif defined(SIMT_CROSS_CORE_U2)
using LaunchState = pa_scheduler::simt_cross_core::u2::U2FullPaState;
#else
using LaunchState = FullPaState;
#endif

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t batches = 256U;
    uint32_t runs = 2U;
    uint32_t builder_count = g0::kDefaultBuilderCount;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    std::string swimlane_json;
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
    std::string acl_config;
#endif
};

constexpr uint32_t kHostTasksPerBatch = 5U;
constexpr uint32_t kHostBuilderWarpCount = g0::kBuilderWarpCount;
constexpr uint32_t kHostWarpThreads = 32U;
constexpr uint32_t kHostBuilderThreadCount = kHostBuilderWarpCount * kHostWarpThreads;
constexpr uint32_t kHostMaxBuilderThreadCount = kHostBuilderThreadCount * g0::kMaxBuilderCount;
constexpr uint32_t kHostPayloadHeaderBytes = 64U;
constexpr uint32_t kHostTensorDescBytes = 128U;
constexpr uint32_t kHostTensorDims = 5U;
constexpr uint32_t kHostOwnerCount = 96U;
constexpr uint32_t kHostWorkloadTileElements = 128U * 128U;
constexpr uint32_t kHostWorkloadInputTiles = 2U;
constexpr uint32_t kHostWorkloadOutputTilesPerOwner = 2U;
constexpr uint32_t kHostWorkloadTiles = kHostWorkloadInputTiles + kHostOwnerCount * kHostWorkloadOutputTilesPerOwner;
constexpr float kHostWorkloadInputA = 2.0F;
constexpr float kHostWorkloadInputB = 3.0F;
constexpr float kHostWorkloadSentinel = -12345.0F;
constexpr uint64_t kHostSyntheticHeapBase = UINT64_C(0x100000000);
constexpr uint64_t kHostSyntheticQueryBase = UINT64_C(0x200000000);
constexpr uint64_t kHostSyntheticKeyBase = UINT64_C(0x300000000);
constexpr uint64_t kHostSyntheticValueBase = UINT64_C(0x400000000);
constexpr uint64_t kHostSyntheticBlockTableBase = UINT64_C(0x500000000);
constexpr uint64_t kHostSyntheticOutputBase = UINT64_C(0x600000000);

static_assert(g0::kTasksPerBatch == kHostTasksPerBatch, "host PA task plan disagrees with the device ABI");
static_assert(
    g0::kBuilderWarpCount == kHostBuilderWarpCount && g0::kWarpSize == kHostWarpThreads,
    "host SIMT mapping disagrees with the device ABI"
);
static_assert(
    g0::kMaxBuilderThreadCount == kHostMaxBuilderThreadCount,
    "host maximum SIMT report capacity disagrees with the device ABI"
);
static_assert(g0::kTensorDescBytes == kHostTensorDescBytes, "host TensorDesc size disagrees with the device ABI");
static_assert(g0::kTokensPerOwner == 4U, "G0 must retain exactly four execution tokens per owner");
static_assert(
    kWorkloadBytes == static_cast<uint64_t>(kHostWorkloadTiles) * kHostWorkloadTileElements * sizeof(float),
    "host workload layout disagrees with the device ABI"
);

enum class HostDataType : uint8_t {
    Float32 = 0U,
    Float16 = 1U,
    Int32 = 2U,
    Bfloat16 = 6U,
};

struct HostTensorDesc {
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint64_t owner_task_id;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    HostDataType dtype;
    bool manual_dep;
    bool is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kHostTensorDims];
    uint64_t extent_elem_cache;
    uint32_t strides[kHostTensorDims];
    uint8_t padding[36U];
};

static_assert(sizeof(HostTensorDesc) == kHostTensorDescBytes, "host TensorDesc ABI must remain 128 bytes");
static_assert(
    offsetof(HostTensorDesc, shapes) == 44U && offsetof(HostTensorDesc, extent_elem_cache) == 64U,
    "host TensorDesc ABI offsets changed"
);

enum class HostTaskKind : uint32_t {
    Alloc = 0U,
    Qk = 1U,
    Sf = 2U,
    Pv = 3U,
    Up = 4U,
};

enum class HostEngine : uint32_t {
    None = 0U,
    Aic = 1U,
    Aiv = 2U,
};

struct HostDecodedExecState {
    uint32_t phase;
    uint32_t build_owner;
    uint32_t execute_owner;
    uint32_t engine;
    uint32_t payload_lines;
    uint32_t task_id;
    bool known_bits;
};

HostDecodedExecState DecodeHostExecState(int64_t signed_raw) {
    constexpr uint64_t kPhaseMask = 0x7ULL;
    constexpr uint64_t kBuildOwnerMask = 0xFFULL;
    constexpr uint64_t kExecuteOwnerMask = 0xFFULL;
    constexpr uint64_t kEngineMask = 0x7ULL;
    constexpr uint64_t kPayloadLinesMask = 0x7FULL;
    constexpr uint64_t kTaskIdMask = 0xFFFFFFFFULL;
    constexpr uint32_t kBuildOwnerShift = 3U;
    constexpr uint32_t kExecuteOwnerShift = 11U;
    constexpr uint32_t kEngineShift = 19U;
    constexpr uint32_t kPayloadLinesShift = 22U;
    constexpr uint32_t kTaskIdShift = 29U;
    constexpr uint64_t kKnownMask = kPhaseMask | (kBuildOwnerMask << kBuildOwnerShift) |
                                    (kExecuteOwnerMask << kExecuteOwnerShift) | (kEngineMask << kEngineShift) |
                                    (kPayloadLinesMask << kPayloadLinesShift) | (kTaskIdMask << kTaskIdShift);
    const uint64_t raw = static_cast<uint64_t>(signed_raw);
    return HostDecodedExecState{
        static_cast<uint32_t>(raw & kPhaseMask),
        static_cast<uint32_t>((raw >> kBuildOwnerShift) & kBuildOwnerMask),
        static_cast<uint32_t>((raw >> kExecuteOwnerShift) & kExecuteOwnerMask),
        static_cast<uint32_t>((raw >> kEngineShift) & kEngineMask),
        static_cast<uint32_t>((raw >> kPayloadLinesShift) & kPayloadLinesMask),
        static_cast<uint32_t>((raw >> kTaskIdShift) & kTaskIdMask),
        (raw & ~kKnownMask) == 0U,
    };
}

struct HostTaskOracle {
    uint32_t task_id;
    uint32_t batch;
    HostTaskKind kind;
    HostEngine engine;
    uint32_t output_count;
    uint64_t output_bytes;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint32_t fanin_count;
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t written_words;
    uint32_t builder_thread;
    std::array<uint64_t, 3U> scalars;
    std::array<int32_t, 3U> predecessors;
};

HostTaskOracle BuildHostTaskOracle(uint32_t task_id) {
    HostTaskOracle task{};
    task.task_id = task_id;
    task.batch = task_id / kHostTasksPerBatch;
    task.kind = static_cast<HostTaskKind>(task_id % kHostTasksPerBatch);
    task.builder_thread = (task_id % kHostBuilderWarpCount) * kHostWarpThreads;
    switch (task.kind) {
    case HostTaskKind::Alloc:
        task.engine = HostEngine::None;
        task.output_count = 3U;
        task.output_bytes = 10240U;
        break;
    case HostTaskKind::Qk:
        task.engine = HostEngine::Aic;
        task.output_count = 1U;
        task.output_bytes = 524288U;
        task.tensor_count = 4U;
        task.scalar_count = 2U;
        task.scalars = {64U, static_cast<uint64_t>(task.batch) * 256U, 0U};
        break;
    case HostTaskKind::Sf:
        task.engine = HostEngine::Aiv;
        task.output_count = 3U;
        task.output_bytes = 264192U;
        task.tensor_count = 4U;
        task.scalar_count = 3U;
        task.fanin_count = 1U;
        task.scalars = {0x3F800000U, 64U, 128U};
        task.predecessors[0] = static_cast<int32_t>(task_id - 1U);
        break;
    case HostTaskKind::Pv:
        task.engine = HostEngine::Aic;
        task.output_count = 1U;
        task.output_bytes = 8192U;
        task.tensor_count = 4U;
        task.scalar_count = 2U;
        task.fanin_count = 1U;
        task.scalars = {64U, static_cast<uint64_t>(task.batch) * 256U, 0U};
        task.predecessors[0] = static_cast<int32_t>(task_id - 1U);
        break;
    case HostTaskKind::Up:
        task.engine = HostEngine::Aiv;
        task.tensor_count = 7U;
        task.scalar_count = 2U;
        task.fanin_count = 3U;
        task.scalars = {1U, 1U, 0U};
        task.predecessors = {
            static_cast<int32_t>(task_id - 2U),
            static_cast<int32_t>(task_id - 1U),
            static_cast<int32_t>(task.batch * kHostTasksPerBatch),
        };
        break;
    }
    if (task.kind != HostTaskKind::Alloc) {
        task.payload_bytes = kHostPayloadHeaderBytes + task.tensor_count * kHostTensorDescBytes +
                             task.scalar_count * sizeof(uint64_t) + task.fanin_count * sizeof(int32_t);
        task.payload_lines = (task.payload_bytes + kHostPayloadHeaderBytes - 1U) / kHostPayloadHeaderBytes;
        task.written_words = kHostPayloadHeaderBytes / sizeof(uint64_t) +
                             task.tensor_count * kHostTensorDescBytes / sizeof(uint64_t) + task.scalar_count +
                             (task.fanin_count + 1U) / 2U;
    }
    return task;
}

struct HostSharedWriterIntent {
    uint32_t producer_task;
    uint32_t output_slot;
};

// Host oracle 独立按 PA 参数 schema 重建 shared writer intent；device 侧只会
// 看到打包后的通用 contract，不能用 TaskKind 直接决定是否进入插入链。
uint32_t HostMetadataWriterIntentCount(uint32_t task_id) {
    return BuildHostTaskOracle(task_id).kind == HostTaskKind::Up ? 3U : 0U;
}

bool HostMetadataWriterIntentAt(uint32_t task_id, uint32_t writer, HostSharedWriterIntent *intent) {
    const HostTaskOracle task = BuildHostTaskOracle(task_id);
    if (intent == nullptr || task.kind != HostTaskKind::Up || writer >= 3U) {
        return false;
    }
    intent->producer_task = task.batch * kHostTasksPerBatch;
    intent->output_slot = 2U - writer;
    return true;
}

uint32_t HostPreviousMetadataWriterTask(uint32_t task_id) {
    if (HostMetadataWriterIntentCount(task_id) == 0U) {
        return UINT32_MAX;
    }
    while (task_id != 0U) {
        --task_id;
        if (HostMetadataWriterIntentCount(task_id) != 0U) {
            return task_id;
        }
    }
    return UINT32_MAX;
}

uint64_t HostMetadataInsertContract(uint32_t task_id) {
    constexpr uint64_t kPresent = uint64_t{1} << 63U;
    constexpr uint32_t kPredecessorShift = 8U;
    const uint32_t writer_count = HostMetadataWriterIntentCount(task_id);
    if (writer_count == 0U) {
        return 0U;
    }
    const uint32_t predecessor = HostPreviousMetadataWriterTask(task_id);
    const uint64_t predecessor_code = predecessor == UINT32_MAX ? 0U : static_cast<uint64_t>(predecessor) + 1U;
    return kPresent | writer_count | (predecessor_code << kPredecessorShift);
}

int32_t HostLatestMetadataWriterBefore(
    uint32_t task_limit, uint32_t producer_task, uint32_t output_slot
) {
    int32_t latest = static_cast<int32_t>(producer_task);
    for (uint32_t task_id = producer_task + 1U; task_id < task_limit; ++task_id) {
        const uint32_t writer_count = HostMetadataWriterIntentCount(task_id);
        for (uint32_t writer = 0U; writer < writer_count; ++writer) {
            HostSharedWriterIntent intent{};
            if (HostMetadataWriterIntentAt(task_id, writer, &intent) &&
                intent.producer_task == producer_task && intent.output_slot == output_slot) {
                latest = static_cast<int32_t>(task_id);
            }
        }
    }
    return latest;
}

uint64_t ExpectedBuilderChecksum(
    uint64_t nonce, uint32_t thread_id, uint32_t task_count, uint32_t wins, uint32_t first_task, uint32_t last_task,
    uint32_t attempts, uint32_t prepares, uint32_t commits, uint32_t insert_waits, uint32_t claim_losses
) {
    uint64_t value = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ task_count;
    value ^= static_cast<uint64_t>(wins) << 1U;
    value ^= static_cast<uint64_t>(first_task) << 7U;
    value ^= static_cast<uint64_t>(last_task) << 19U;
    value ^= static_cast<uint64_t>(attempts) << 37U;
    value ^= static_cast<uint64_t>(prepares) << 43U;
    value ^= static_cast<uint64_t>(commits) << 49U;
    value ^= static_cast<uint64_t>(insert_waits) << 55U;
    value ^= static_cast<uint64_t>(claim_losses) << 25U;
    value *= UINT64_C(0x9E3779B97F4A7C15);
    value ^= value >> 29U;
    value *= UINT64_C(0xD6E8FEB86659FD93);
    return value ^ (value >> 31U);
}

uint32_t ExpectedExecutionTaskId(HostEngine engine, uint32_t ticket_index) {
    const uint32_t batch = ticket_index / 2U;
    const uint32_t within_batch = ticket_index % 2U;
    if (engine == HostEngine::Aic) {
        return batch * kHostTasksPerBatch + (within_batch == 0U ? 1U : 3U);
    }
    return batch * kHostTasksPerBatch + (within_batch == 0U ? 2U : 4U);
}

uint64_t ExpectedHeapBytes(uint32_t batches) {
    uint64_t total = 0U;
    const uint32_t task_count = batches * kHostTasksPerBatch;
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        total += BuildHostTaskOracle(task_id).output_bytes;
    }
    return total;
}

uint64_t ExpectedHeapShardBytes(uint32_t batches, uint32_t shard) {
    uint64_t total = 0U;
    const uint32_t task_count = batches * kHostTasksPerBatch;
    for (uint32_t task_id = shard; task_id < task_count; task_id += 8U) {
        total += BuildHostTaskOracle(task_id).output_bytes;
    }
    return total;
}

uint64_t HostElementBytes(HostDataType dtype) {
    switch (dtype) {
    case HostDataType::Float32:
    case HostDataType::Int32:
        return 4U;
    case HostDataType::Float16:
    case HostDataType::Bfloat16:
        return 2U;
    }
    return 0U;
}

HostTensorDesc MakeDenseTensor(
    uint64_t address, uint64_t owner, HostDataType dtype, uint32_t ndims, uint32_t shape0, uint32_t shape1
) {
    HostTensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.owner_task_id = owner;
    tensor.ndims = ndims;
    tensor.dtype = dtype;
    tensor.is_contiguous = true;
    tensor.shapes[0] = shape0;
    tensor.strides[0] = ndims == 1U ? 1U : shape1;
    tensor.extent_elem_cache = shape0;
    if (ndims == 2U) {
        tensor.shapes[1] = shape1;
        tensor.strides[1] = 1U;
        tensor.extent_elem_cache *= shape1;
    }
    tensor.buffer_size = tensor.extent_elem_cache * HostElementBytes(dtype);
    return tensor;
}

HostTensorDesc MakeOutputTensor(const HostTaskOracle &task, uint32_t slot, uint64_t task_base) {
    uint64_t offset = 0U;
    uint32_t ndims = 0U;
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    HostDataType dtype = HostDataType::Float32;
    switch (task.kind) {
    case HostTaskKind::Alloc:
        if (slot == 0U) {
            ndims = 2U;
            shape0 = 16U;
            shape1 = 128U;
        } else if (slot == 1U || slot == 2U) {
            offset = slot == 1U ? 8192U : 9216U;
            ndims = 1U;
            shape0 = 16U;
        }
        break;
    case HostTaskKind::Qk:
        if (slot == 0U) {
            ndims = 2U;
            shape0 = 16U;
            shape1 = 8192U;
        }
        break;
    case HostTaskKind::Sf:
        if (slot == 0U) {
            ndims = 2U;
            shape0 = 16U;
            shape1 = 8192U;
            dtype = HostDataType::Bfloat16;
        } else if (slot == 1U || slot == 2U) {
            offset = slot == 1U ? 262144U : 263168U;
            ndims = 1U;
            shape0 = 16U;
        }
        break;
    case HostTaskKind::Pv:
        if (slot == 0U) {
            ndims = 2U;
            shape0 = 16U;
            shape1 = 128U;
        }
        break;
    case HostTaskKind::Up:
        break;
    }
    return MakeDenseTensor(kHostSyntheticHeapBase + task_base + offset, task.task_id, dtype, ndims, shape0, shape1);
}

HostTensorDesc MakeExternalTensor(uint64_t address, HostDataType dtype, uint32_t shape0, uint32_t shape1) {
    return MakeDenseTensor(address, UINT64_MAX, dtype, 2U, shape0, shape1);
}

HostTensorDesc MakeQueryView(uint32_t batches, uint32_t batch) {
    HostTensorDesc tensor = MakeExternalTensor(kHostSyntheticQueryBase, HostDataType::Bfloat16, batches * 16U, 128U);
    tensor.start_offset = static_cast<uint64_t>(batch) * 16U * 128U;
    tensor.shapes[0] = 16U;
    tensor.extent_elem_cache = 16U * 128U;
    return tensor;
}

HostTensorDesc MakeOutputView(uint32_t batches, uint32_t batch) {
    HostTensorDesc tensor = MakeExternalTensor(kHostSyntheticOutputBase, HostDataType::Float32, batches * 16U, 128U);
    tensor.start_offset = static_cast<uint64_t>(batch) * 16U * 128U;
    tensor.manual_dep = true;
    tensor.shapes[0] = 16U;
    tensor.extent_elem_cache = 16U * 128U;
    return tensor;
}

uint64_t HostGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = UINT64_C(0x4755415244473041) ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 29U;
    value *= UINT64_C(0x9E3779B97F4A7C15);
    return value ^ (value >> 31U);
}

uint64_t HostDescriptorPoisonWord(uint64_t nonce, uint32_t task_id, uint32_t slot, uint32_t word) {
    return UINT64_C(0xB4B4B4B4B4B4B4B4) ^ nonce ^ (static_cast<uint64_t>(task_id) << 32U) ^
           (static_cast<uint64_t>(slot) << 16U) ^ word;
}

uint64_t HostPayloadPoisonWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    return UINT64_C(0xA5A5A5A5A5A5A5A5) ^ nonce ^ (static_cast<uint64_t>(task_id) << 32U) ^ word;
}

void InitializeGuard(g0::FullPaGuard *guard, uint64_t nonce, uint32_t guard_id) {
    for (uint32_t word = 0U; word < g0::kCacheLineBytes / sizeof(uint64_t); ++word) {
        guard->words[word] = HostGuardWord(nonce, guard_id, word);
    }
}

void InitializeAtomic(g0::AtomicLine *line, int64_t value) { line->value = value; }

void InitializeToken(g0::ExecutionToken *token) {
    token->control.phase = g0::ExecTokenPhase::Idle;
    token->control.task_id = UINT32_MAX;
    token->control.build_owner = UINT32_MAX;
    token->control.execute_owner = UINT32_MAX;
    token->control.engine_class = g0::ExecEngineClass::None;
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.fanin_ready_prefix = 0U;
    token->control.payload_address = 0U;
    token->control.completion_vend = 0U;
    token->control.function_and_reference = 0U;
    token->control.shape_and_scalar_offset = 0U;
}

void InitializeState(
    FullPaState *state, uint64_t nonce, uint32_t batches, uint32_t builder_count, uint64_t workspace_address
) {
    std::memset(state, 0xA5, sizeof(*state));
    state->control.magic = g0::kProbeMagic;
    state->control.version = g0::kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = UINT64_C(1000000000);
    state->control.batch_count = batches;
    state->control.task_count = batches * kHostTasksPerBatch;
    state->control.kernel_task_count = batches * 4U;
    state->control.builder_thread_count = kHostBuilderThreadCount;
    state->control.heap_base = kHostSyntheticHeapBase;
    state->control.heap_bytes = UINT64_C(256) << 20U;
    state->control.workspace_base = workspace_address;
    state->control.workspace_bytes = kWorkloadBytes;
    state->control.qk_repeats = 1U;
    state->control.sf_repeats = 1U;
    state->control.pv_repeats = 1U;
    state->control.up_repeats = 1U;
    state->control.builder_count = builder_count;
    state->control.reserved32 = 0U;
    for (uint64_t &reserved : state->control.reserved) {
        reserved = 0U;
    }

    g0::FullPaGuard *guards[] = {
        &state->guard_before_tasks,           &state->guard_after_tasks,           &state->guard_before_tokens,
        &state->guard_after_tokens,           &state->guard_before_roles,          &state->guard_after_roles,
        &state->guard_before_builder_threads, &state->guard_after_builder_threads,
    };
    for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
        InitializeGuard(guards[guard], nonce, guard);
    }

    const uint32_t task_count = batches * kHostTasksPerBatch;
    for (uint32_t task_id = 0U; task_id < g0::kMaxTasks; ++task_id) {
        g0::FullPaTask &task = state->tasks[task_id];
        std::memset(&task.plan, 0xD3, sizeof(task.plan));
        task.completion.flag = 0;
        task.completion.vend = 0U;
        task.completion.deps_prepared = static_cast<int64_t>(task_id) - 1;
        InitializeAtomic(&task.insert_completion, static_cast<int64_t>(task_id) - 1);
        InitializeAtomic(&task.allocation.task_base_plus_one, 0);
        InitializeAtomic(&task.allocation.completion_vend_plus_one, 0);
        for (uint32_t slot = 0U; slot < g0::kOutputsPerTask; ++slot) {
            uint64_t *published_words = reinterpret_cast<uint64_t *>(&task.outputs.published[0]);
            uint64_t *last_writer_words = reinterpret_cast<uint64_t *>(&task.outputs.last_writer[0]);
            published_words[slot] = static_cast<uint64_t>(-1);
            last_writer_words[slot] = static_cast<uint64_t>(-1);
            std::memset(&task.outputs.tensors[slot], 0, sizeof(task.outputs.tensors[slot]));
        }
        if (task_id < task_count) {
            const uint32_t active_outputs = BuildHostTaskOracle(task_id).output_count;
            for (uint32_t slot = 0U; slot < active_outputs; ++slot) {
                uint64_t representation[g0::kTensorDescWords] = {};
                for (uint32_t word = 0U; word < g0::kTensorDescWords; ++word) {
                    representation[word] = HostDescriptorPoisonWord(nonce, task_id, slot, word);
                }
                std::memcpy(&task.outputs.tensors[slot], representation, sizeof(representation));
            }
        }
        task.exec.control.state = 0;
        for (uint32_t word = 0U; word < g0::kMaxPayloadWords; ++word) {
            task.exec.payload.words[word] = HostPayloadPoisonWord(nonce, task_id, word);
        }
        std::memset(&task.build_report, 0xD3, sizeof(task.build_report));
        if (task_id < task_count) {
            task.build_report.build_attempt_count = 0U;
            task.build_report.build_win_count = 0U;
        }
        std::memset(&task.execution_witness, 0xD3, sizeof(task.execution_witness));
        if (task_id < task_count && BuildHostTaskOracle(task_id).engine != HostEngine::None) {
            task.execution_witness.state = 0;
        }
    }

    state->fatal.state = 0;
    InitializeAtomic(&state->drain.builder_started, 0);
    InitializeAtomic(&state->drain.builder_finished, 0);
    InitializeAtomic(&state->drain.done_count, 0);
    InitializeAtomic(&state->drain.alloc_done, 0);
    InitializeAtomic(&state->drain.aic_done, 0);
    InitializeAtomic(&state->drain.aiv_done, 0);
    InitializeAtomic(&state->drain.root_finished, 0);
    for (uint32_t group = 0U; group < g0::kDrainGroupCount; ++group) {
        InitializeAtomic(&state->drain.arrivals[group], 0);
    }

    InitializeAtomic(&state->exec_dispatch.aic_next, 0);
    InitializeAtomic(&state->exec_dispatch.aiv_next, 0);
    state->exec_dispatch.aic_task_count = batches * 2U;
    state->exec_dispatch.aiv_task_count = batches * 2U;
    for (uint32_t index = 0U; index < batches * 2U; ++index) {
        state->exec_dispatch.aic_task_ids[index] = ExpectedExecutionTaskId(HostEngine::Aic, index);
        state->exec_dispatch.aiv_task_ids[index] = ExpectedExecutionTaskId(HostEngine::Aiv, index);
    }
    for (uint32_t shard = 0U; shard < g0::kSharedHeapShards; ++shard) {
        InitializeAtomic(&state->heap.shard_cursors[shard], 0);
    }
    InitializeAtomic(&state->heap.aggregate_vend, 0);
    InitializeAtomic(&state->ordinary_map.head, -1);
    InitializeAtomic(&state->ordinary_map.tail, -1);
    InitializeAtomic(&state->ordinary_map.lookup_count, -1);
    InitializeAtomic(&state->ordinary_map.append_count, -1);

    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        for (uint32_t slot = 0U; slot < g0::kTokensPerOwner; ++slot) {
            InitializeToken(&state->tokens[owner][slot]);
        }
        std::memset(&state->roles[owner], 0xD3, sizeof(state->roles[owner]));
    }
    for (uint32_t thread = 0U; thread < g0::kMaxBuilderThreadCount; ++thread) {
        std::memset(&state->builder_threads[thread], 0xD3, sizeof(state->builder_threads[thread]));
    }
}

#if defined(SIMT_CROSS_CORE_U2)
void InitializeU2Guard(u2::U2Guard *guard, uint64_t nonce, uint32_t guard_id) {
    for (uint32_t word = 0U; word < ubuf_staging::kWordsPerLine; ++word) {
        guard->words[word] = u2::ExpectedGuardWord(nonce, guard_id, word);
    }
}

void InitializeU2Staging(u2::U2StagingState *staging, uint64_t nonce, uint32_t batches) {
    std::memset(staging, 0xA5, sizeof(*staging));
    staging->control.magic = u2::kProbeMagic;
    staging->control.version = u2::kProbeVersion;
    staging->control.launch_nonce = nonce;
    staging->control.batch_count = batches;
    staging->control.task_count = g0::TaskCount(batches);
    staging->control.kernel_task_count = g0::KernelTaskCount(batches);
    staging->control.builder_count = u2::kBuilderCount;
    staging->control.slot_count = ubuf_staging::kSlotCount;
    staging->control.max_payload_lines = ubuf_staging::kMaxPayloadLines;
    staging->control.words_per_line = ubuf_staging::kWordsPerLine;
    staging->control.alignment_bytes = ubuf_staging::kAlignmentBytes;
    staging->control.transport_kind = ubuf_staging::TransportKind::SimtUbufReadToGmWordStore;
    staging->control.reserved16 = 0U;
    staging->control.slot_stride_bytes = ubuf_staging::kSlotStrideBytes;
    staging->control.region_bytes = ubuf_staging::kRegionBytes;
    staging->control.payload_offset_bytes = ubuf_staging::kPayloadOffsetBytes;

    InitializeU2Guard(&staging->guard_before_slots, nonce, u2::kGuardBeforeSlots);
    InitializeU2Guard(&staging->guard_after_slots, nonce, u2::kGuardAfterSlots);
    InitializeU2Guard(&staging->guard_before_reports, nonce, u2::kGuardBeforeReports);
    InitializeU2Guard(&staging->guard_after_reports, nonce, u2::kGuardAfterReports);
    for (uint32_t slot = 0U; slot < ubuf_staging::kSlotCount; ++slot) {
        InitializeAtomic(&staging->slot_states[slot], static_cast<int64_t>(ubuf_staging::SlotFreeState(0U)));
        InitializeAtomic(&staging->slot_acquire_count[slot], 0);
        InitializeAtomic(&staging->slot_release_count[slot], 0);
    }
    InitializeAtomic(&staging->global_busy_depth, 0);
    InitializeAtomic(&staging->global_max_busy_depth, 0);
    InitializeAtomic(&staging->anchor_staged_count, 0);
    InitializeAtomic(&staging->anchor_staged_mask, 0);
    InitializeAtomic(&staging->guard_check_count, 0);
    InitializeAtomic(&staging->ubuf_words_written, 0);
    InitializeAtomic(&staging->gm_words_stored, 0);
    std::memset(staging->reports, 0xD3, sizeof(staging->reports));
}
#endif

#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
void InitializeSwimlaneTrace(g0_swimlane::TraceState *trace, uint64_t nonce, uint32_t batches, uint32_t builder_count) {
    std::memset(trace, 0xD3, sizeof(*trace));
    trace->control.magic = g0_swimlane::kTraceMagic;
    trace->control.version = g0_swimlane::kTraceVersion;
    trace->control.launch_nonce = nonce;
    trace->control.tick_ns = 1U;
    trace->control.task_count = g0::TaskCount(batches);
    trace->control.kernel_task_count = g0::KernelTaskCount(batches);
    trace->control.builder_count = builder_count;
    trace->control.role_count = g0::kOwnerCount;
    trace->control.simt_writer_count = g0::BuilderLeaderCount(builder_count);
    trace->control.scalar_writer_count = g0_swimlane::kTraceScalarWriterCount;
    trace->control.simt_records_per_writer = g0_swimlane::kTraceSimtRecordsPerWriter;
    trace->control.scalar_records_per_writer = g0_swimlane::kTraceScalarRecordsPerWriter;
    trace->control.record_size_bytes = sizeof(g0_swimlane::TraceRecord);
    trace->control.reserved32 = 0U;
    std::memset(trace->control.reserved, 0, sizeof(trace->control.reserved));
}
#endif

FullPaState &FullPaView(LaunchState &state) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
    return state.full_pa;
#else
    return state;
#endif
}

const FullPaState &FullPaView(const LaunchState &state) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
    return state.full_pa;
#else
    return state;
#endif
}

void InitializeLaunchState(
    LaunchState *state, uint64_t nonce, uint32_t batches, uint32_t builder_count, uint64_t workspace_address
) {
    InitializeState(&FullPaView(*state), nonce, batches, builder_count, workspace_address);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    InitializeSwimlaneTrace(&state->trace, nonce, batches, builder_count);
#elif defined(SIMT_CROSS_CORE_U2)
    InitializeU2Staging(&state->staging, nonce, batches);
#endif
}

bool ExpectedPayloadTensor(
    const HostTaskOracle &task, uint32_t tensor_index, uint32_t batches, const std::vector<uint64_t> &task_bases,
    HostTensorDesc *expected
) {
    if (expected == nullptr || task.task_id >= task_bases.size()) {
        return false;
    }
    const auto output = [&task_bases](uint32_t producer, uint32_t slot, HostTensorDesc *tensor) {
        if (producer >= task_bases.size()) {
            return false;
        }
        *tensor = MakeOutputTensor(BuildHostTaskOracle(producer), slot, task_bases[producer]);
        return true;
    };
    switch (task.kind) {
    case HostTaskKind::Qk:
        if (tensor_index == 0U) {
            *expected = MakeQueryView(batches, task.batch);
            return true;
        }
        if (tensor_index == 1U) {
            *expected = MakeExternalTensor(kHostSyntheticKeyBase, HostDataType::Bfloat16, batches * 64U * 128U, 128U);
            return true;
        }
        if (tensor_index == 2U) {
            *expected = MakeExternalTensor(kHostSyntheticBlockTableBase, HostDataType::Int32, batches, 256U);
            return true;
        }
        return tensor_index == 3U && output(task.task_id, 0U, expected);
    case HostTaskKind::Sf:
        if (tensor_index == 0U) {
            return output(task.task_id - 1U, 0U, expected);
        }
        return tensor_index >= 1U && tensor_index <= 3U && output(task.task_id, tensor_index - 1U, expected);
    case HostTaskKind::Pv:
        if (tensor_index == 0U) {
            return output(task.task_id - 1U, 0U, expected);
        }
        if (tensor_index == 1U) {
            *expected = MakeExternalTensor(kHostSyntheticValueBase, HostDataType::Bfloat16, batches * 64U * 128U, 128U);
            return true;
        }
        if (tensor_index == 2U) {
            *expected = MakeExternalTensor(kHostSyntheticBlockTableBase, HostDataType::Int32, batches, 256U);
            return true;
        }
        return tensor_index == 3U && output(task.task_id, 0U, expected);
    case HostTaskKind::Up:
        if (tensor_index == 0U || tensor_index == 1U) {
            return output(task.task_id - 2U, tensor_index + 1U, expected);
        }
        if (tensor_index == 2U) {
            return output(task.task_id - 1U, 0U, expected);
        }
        if (tensor_index >= 3U && tensor_index <= 5U) {
            return output(task.batch * kHostTasksPerBatch, 5U - tensor_index, expected);
        }
        if (tensor_index == 6U) {
            *expected = MakeOutputView(batches, task.batch);
            return true;
        }
        return false;
    case HostTaskKind::Alloc:
        return false;
    }
    return false;
}

struct ValidationFailure {
    const char *reason = "none";
    uint32_t task = UINT32_MAX;
    uint32_t owner = UINT32_MAX;
    uint32_t index = UINT32_MAX;
    uint64_t expected = 0U;
    uint64_t actual = 0U;
};

bool RecordCheck(
    bool condition, ValidationFailure *failure, const char *reason, uint32_t task = UINT32_MAX,
    uint32_t owner = UINT32_MAX, uint32_t index = UINT32_MAX, uint64_t expected = 0U, uint64_t actual = 0U
) {
    if (condition) {
        return true;
    }
    if (std::strcmp(failure->reason, "none") == 0) {
        failure->reason = reason;
        failure->task = task;
        failure->owner = owner;
        failure->index = index;
        failure->expected = expected;
        failure->actual = actual;
    }
    return false;
}

HostTensorDesc LoadPayloadTensor(const g0::SharedExecCell &cell, uint32_t tensor_index) {
    HostTensorDesc tensor{};
    uint64_t representation[g0::kTensorDescWords] = {};
    const uint32_t first_word = g0::kPayloadHeaderWords + tensor_index * g0::kTensorDescWords;
    for (uint32_t word = 0U; word < g0::kTensorDescWords; ++word) {
        representation[word] = cell.payload.words[first_word + word];
    }
    std::memcpy(&tensor, representation, sizeof(tensor));
    return tensor;
}

uint64_t HostExecutionWitnessState(uint64_t nonce, uint32_t task_id, HostTaskKind kind, uint32_t owner) {
    uint64_t value = UINT64_C(0x5749544E45535330) ^ nonce ^ (static_cast<uint64_t>(task_id) << 17U) ^
                     (static_cast<uint64_t>(kind) << 49U) ^ (static_cast<uint64_t>(owner) << 1U);
    value ^= value >> 23U;
    value *= UINT64_C(0xD6E8FEB86659FD93);
    value ^= value >> 32U;
    return value == 0U ? 1U : value;
}

uint64_t ExpectedWitnessChecksum(HostTaskKind kind) {
    uint32_t bits = 0U;
    switch (kind) {
    case HostTaskKind::Qk:
    case HostTaskKind::Pv:
        bits = 0x44400000U;
        break;
    case HostTaskKind::Sf:
        bits = 0x40A00000U;
        break;
    case HostTaskKind::Up:
        bits = 0x40C00000U;
        break;
    case HostTaskKind::Alloc:
        break;
    }
    return static_cast<uint64_t>(bits) | (static_cast<uint64_t>(bits) << 32U);
}

bool AtomicPaddingMatches(const g0::AtomicLine &actual, const g0::AtomicLine &initial) {
    return std::memcmp(actual.padding, initial.padding, sizeof(actual.padding)) == 0;
}

#if defined(SIMT_CROSS_CORE_U2)
bool ValidateU2Atomic(
    const g0::AtomicLine &actual, const g0::AtomicLine &initial, int64_t expected_value, const char *value_reason,
    const char *padding_reason, uint32_t index, ValidationFailure *failure
) {
    return RecordCheck(
               actual.value == expected_value, failure, value_reason, UINT32_MAX, UINT32_MAX, index,
               static_cast<uint64_t>(expected_value), static_cast<uint64_t>(actual.value)
           ) &&
           RecordCheck(AtomicPaddingMatches(actual, initial), failure, padding_reason, UINT32_MAX, UINT32_MAX, index);
}

bool ValidateU2Guard(const u2::U2Guard &actual, uint64_t nonce, uint32_t guard_id, ValidationFailure *failure) {
    for (uint32_t word = 0U; word < ubuf_staging::kWordsPerLine; ++word) {
        const uint64_t expected = u2::ExpectedGuardWord(nonce, guard_id, word);
        if (!RecordCheck(
                actual.words[word] == expected, failure, "u2-guard-mutated", UINT32_MAX, UINT32_MAX,
                guard_id * ubuf_staging::kWordsPerLine + word, expected, actual.words[word]
            )) {
            return false;
        }
    }
    return true;
}

uint64_t ExpectedU2PayloadChecksum(const FullPaState &state, uint64_t nonce, uint32_t task_id, uint32_t words) {
    uint64_t checksum = u2::PayloadChecksumSeed(nonce, task_id, words);
    for (uint32_t word = 0U; word < words; ++word) {
        checksum = u2::FoldPayloadChecksum(checksum, state.tasks[task_id].exec.payload.words[word]);
    }
    return checksum;
}

bool ValidateU2Staging(
    const LaunchState &state, const LaunchState &initial, uint64_t nonce, uint32_t batches, ValidationFailure *failure
) {
    const u2::U2StagingState &staging = state.staging;
    const u2::U2StagingState &initial_staging = initial.staging;
    if (!RecordCheck(
            std::memcmp(&staging.control, &initial_staging.control, sizeof(staging.control)) == 0, failure,
            "u2-control-mutated"
        ) ||
        !ValidateU2Guard(staging.guard_before_slots, nonce, u2::kGuardBeforeSlots, failure) ||
        !ValidateU2Guard(staging.guard_after_slots, nonce, u2::kGuardAfterSlots, failure) ||
        !ValidateU2Guard(staging.guard_before_reports, nonce, u2::kGuardBeforeReports, failure) ||
        !ValidateU2Guard(staging.guard_after_reports, nonce, u2::kGuardAfterReports, failure)) {
        return false;
    }

    for (uint32_t slot = 0U; slot < ubuf_staging::kSlotCount; ++slot) {
        if (!ValidateU2Atomic(
                staging.slot_states[slot], initial_staging.slot_states[slot],
                static_cast<int64_t>(ubuf_staging::SlotFreeState(batches)), "u2-slot-final-state",
                "u2-slot-padding-mutated", slot, failure
            ) ||
            !ValidateU2Atomic(
                staging.slot_acquire_count[slot], initial_staging.slot_acquire_count[slot], batches,
                "u2-slot-acquire-count", "u2-slot-acquire-padding-mutated", slot, failure
            ) ||
            !ValidateU2Atomic(
                staging.slot_release_count[slot], initial_staging.slot_release_count[slot], batches,
                "u2-slot-release-count", "u2-slot-release-padding-mutated", slot, failure
            )) {
            return false;
        }
    }

    const uint64_t kernel_tasks = g0::KernelTaskCount(batches);
    const uint64_t expected_words = u2::ExpectedWordsPerBatch() * batches;
    struct AtomicExpectation {
        const g0::AtomicLine *actual;
        const g0::AtomicLine *initial;
        int64_t expected;
        const char *reason;
    };
    const AtomicExpectation atomics[] = {
        {&staging.global_busy_depth, &initial_staging.global_busy_depth, 0, "u2-global-busy-depth"},
        {&staging.global_max_busy_depth, &initial_staging.global_max_busy_depth,
         static_cast<int64_t>(ubuf_staging::kSlotCount), "u2-global-max-busy-depth"},
        {&staging.anchor_staged_count, &initial_staging.anchor_staged_count, static_cast<int64_t>(u2::kAnchorTaskCount),
         "u2-anchor-count"},
        {&staging.anchor_staged_mask, &initial_staging.anchor_staged_mask, static_cast<int64_t>(u2::kAnchorMask),
         "u2-anchor-mask"},
        {&staging.guard_check_count, &initial_staging.guard_check_count, static_cast<int64_t>(kernel_tasks),
         "u2-guard-check-count"},
        {&staging.ubuf_words_written, &initial_staging.ubuf_words_written, static_cast<int64_t>(expected_words),
         "u2-ubuf-words-written"},
        {&staging.gm_words_stored, &initial_staging.gm_words_stored, static_cast<int64_t>(expected_words),
         "u2-gm-words-stored"},
    };
    for (uint32_t index = 0U; index < sizeof(atomics) / sizeof(atomics[0]); ++index) {
        if (!ValidateU2Atomic(
                *atomics[index].actual, *atomics[index].initial, atomics[index].expected, atomics[index].reason,
                "u2-counter-padding-mutated", index, failure
            )) {
            return false;
        }
    }

    const uint32_t report_count = g0::KernelTaskCount(batches);
    for (uint32_t report_index = 0U; report_index < report_count; ++report_index) {
        const uint32_t task_id =
            (report_index / g0::kKernelsPerBatch) * g0::kTasksPerBatch + report_index % g0::kKernelsPerBatch + 1U;
        const uint32_t expected_words_for_task = u2::PayloadWrittenWords(g0::TaskKindAt(task_id));
        const u2::U2TaskStagingReport &report = staging.reports[report_index];
        const uint64_t expected_checksum =
            ExpectedU2PayloadChecksum(state.full_pa, nonce, task_id, expected_words_for_task);
        if (!RecordCheck(
                report.task_id == task_id, failure, "u2-report-task", task_id, UINT32_MAX, report_index, task_id,
                report.task_id
            ) ||
            !RecordCheck(
                report.slot_id == u2::SlotForTask(task_id), failure, "u2-report-slot", task_id, UINT32_MAX,
                report_index, u2::SlotForTask(task_id), report.slot_id
            ) ||
            !RecordCheck(
                report.generation == u2::ExpectedGeneration(task_id), failure, "u2-report-generation", task_id,
                UINT32_MAX, report_index, u2::ExpectedGeneration(task_id), report.generation
            ) ||
            !RecordCheck(
                report.phase_bits == u2::kExpectedTransportPhaseBits, failure, "u2-report-phase", task_id, UINT32_MAX,
                report_index, u2::kExpectedTransportPhaseBits, report.phase_bits
            ) ||
            !RecordCheck(
                report.ubuf_words_written == expected_words_for_task, failure, "u2-report-ubuf-words", task_id,
                UINT32_MAX, report_index, expected_words_for_task, report.ubuf_words_written
            ) ||
            !RecordCheck(
                report.gm_words_stored == expected_words_for_task, failure, "u2-report-gm-words", task_id, UINT32_MAX,
                report_index, expected_words_for_task, report.gm_words_stored
            ) ||
            !RecordCheck(
                report.guard_check_count == 1U && report.acquire_count == 1U && report.release_count == 1U, failure,
                "u2-report-exact-counts", task_id, UINT32_MAX, report_index, UINT64_C(0x0000000100010001),
                (static_cast<uint64_t>(report.guard_check_count) << 32U) |
                    (static_cast<uint64_t>(report.acquire_count) << 16U) | report.release_count
            ) ||
            !RecordCheck(
                report.reserved32[0] == 0U && report.reserved32[1] == 0U && report.reserved32[2] == 0U, failure,
                "u2-report-reserved", task_id, UINT32_MAX, report_index
            ) ||
            !RecordCheck(
                report.launch_nonce == nonce, failure, "u2-report-nonce", task_id, UINT32_MAX, report_index, nonce,
                report.launch_nonce
            ) ||
            !RecordCheck(
                report.payload_checksum == expected_checksum, failure, "u2-report-payload-checksum", task_id,
                UINT32_MAX, report_index, expected_checksum, report.payload_checksum
            )) {
            return false;
        }
    }
    for (uint32_t report_index = report_count; report_index < u2::kMaxTransportReports; ++report_index) {
        if (!RecordCheck(
                std::memcmp(
                    &staging.reports[report_index], &initial_staging.reports[report_index],
                    sizeof(staging.reports[report_index])
                ) == 0,
                failure, "u2-inactive-report-mutated", UINT32_MAX, UINT32_MAX, report_index
            )) {
            return false;
        }
    }
    return true;
}
#endif

bool ValidateFixedRegions(
    const FullPaState &state, const FullPaState &initial, uint64_t nonce, uint32_t batches, ValidationFailure *failure
) {
    if (!RecordCheck(
            std::memcmp(&state.control, &initial.control, sizeof(state.control)) == 0, failure, "control-mutated"
        )) {
        return false;
    }
    const g0::FullPaGuard *guards[] = {
        &state.guard_before_tasks,           &state.guard_after_tasks,           &state.guard_before_tokens,
        &state.guard_after_tokens,           &state.guard_before_roles,          &state.guard_after_roles,
        &state.guard_before_builder_threads, &state.guard_after_builder_threads,
    };
    for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
        for (uint32_t word = 0U; word < g0::kCacheLineBytes / sizeof(uint64_t); ++word) {
            if (!RecordCheck(
                    guards[guard]->words[word] == HostGuardWord(nonce, guard, word), failure, "guard-mutated",
                    UINT32_MAX, UINT32_MAX, guard * 8U + word, HostGuardWord(nonce, guard, word),
                    guards[guard]->words[word]
                )) {
                return false;
            }
        }
    }
    if (!RecordCheck(
            state.fatal.state == 0, failure, "fatal-nonzero", UINT32_MAX, UINT32_MAX, UINT32_MAX, 0U,
            static_cast<uint64_t>(state.fatal.state)
        ) ||
        !RecordCheck(
            std::memcmp(state.fatal.padding, initial.fatal.padding, sizeof(state.fatal.padding)) == 0, failure,
            "fatal-padding-mutated"
        ) ||
        !RecordCheck(
            std::memcmp(&state.ordinary_map, &initial.ordinary_map, sizeof(state.ordinary_map)) == 0, failure,
            "ordinary-map-canary-mutated"
        )) {
        return false;
    }
    const uint32_t task_count = batches * kHostTasksPerBatch;
    for (uint32_t task_id = task_count; task_id < g0::kMaxTasks; ++task_id) {
        if (!RecordCheck(
                std::memcmp(&state.tasks[task_id], &initial.tasks[task_id], sizeof(state.tasks[task_id])) == 0, failure,
                "inactive-task-tail-mutated", task_id
            )) {
            return false;
        }
    }
    return true;
}

struct HeapInterval {
    uint64_t begin;
    uint64_t end;
    uint32_t task_id;
};

bool ValidateHeapAndCollectBases(
    const FullPaState &state, const FullPaState &initial, uint32_t batches, std::vector<uint64_t> *task_bases,
    ValidationFailure *failure
) {
    const uint32_t task_count = batches * kHostTasksPerBatch;
    const uint64_t expected_total = ExpectedHeapBytes(batches);
    std::array<std::vector<HeapInterval>, 8U> intervals;
    std::vector<HeapInterval> aggregate_intervals;
    task_bases->assign(task_count, 0U);
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const HostTaskOracle task = BuildHostTaskOracle(task_id);
        const int64_t raw_base = state.tasks[task_id].allocation.task_base_plus_one.value;
        const int64_t raw_vend = state.tasks[task_id].allocation.completion_vend_plus_one.value;
        if (!RecordCheck(
                raw_base > 0, failure, "task-base-report-missing", task_id, UINT32_MAX, UINT32_MAX, 1U,
                static_cast<uint64_t>(raw_base)
            ) ||
            !RecordCheck(
                raw_vend > 0, failure, "task-vend-report-missing", task_id, UINT32_MAX, UINT32_MAX, 1U,
                static_cast<uint64_t>(raw_vend)
            ) ||
            !RecordCheck(
                AtomicPaddingMatches(
                    state.tasks[task_id].allocation.task_base_plus_one,
                    initial.tasks[task_id].allocation.task_base_plus_one
                ),
                failure, "task-base-report-padding-mutated", task_id
            ) ||
            !RecordCheck(
                AtomicPaddingMatches(
                    state.tasks[task_id].allocation.completion_vend_plus_one,
                    initial.tasks[task_id].allocation.completion_vend_plus_one
                ),
                failure, "task-vend-report-padding-mutated", task_id
            )) {
            return false;
        }
        const uint64_t task_base = static_cast<uint64_t>(raw_base - 1);
        const uint64_t vend = static_cast<uint64_t>(raw_vend - 1);
        (*task_bases)[task_id] = task_base;
        if (!RecordCheck(
                task_base % 1024U == 0U, failure, "task-base-misaligned", task_id, UINT32_MAX, UINT32_MAX, 0U,
                task_base % 1024U
            ) ||
            !RecordCheck(
                vend <= expected_total, failure, "task-vend-above-final", task_id, UINT32_MAX, UINT32_MAX,
                expected_total, vend
            ) ||
            !RecordCheck(
                vend % 1024U == 0U, failure, "task-vend-misaligned", task_id, UINT32_MAX, UINT32_MAX, 0U, vend % 1024U
            )) {
            return false;
        }
        if (task.output_bytes == 0U) {
            if (!RecordCheck(
                    task_base == 0U, failure, "zero-output-task-base-nonzero", task_id, UINT32_MAX, UINT32_MAX, 0U,
                    task_base
                )) {
                return false;
            }
            continue;
        }
        if (!RecordCheck(
                vend >= task.output_bytes, failure, "task-vend-below-reservation", task_id, UINT32_MAX, UINT32_MAX,
                task.output_bytes, vend
            )) {
            return false;
        }
        aggregate_intervals.push_back(HeapInterval{vend - task.output_bytes, vend, task_id});
        const uint32_t shard = task_id & 7U;
        const uint64_t shard_begin = static_cast<uint64_t>(shard) * (UINT64_C(32) << 20U);
        const uint64_t shard_end = shard_begin + (UINT64_C(32) << 20U);
        if (!RecordCheck(
                task_base >= shard_begin && task_base <= shard_end && task.output_bytes <= shard_end - task_base,
                failure, "task-heap-interval-outside-shard", task_id, UINT32_MAX, shard, shard_begin, task_base
            )) {
            return false;
        }
        intervals[shard].push_back(HeapInterval{task_base, task_base + task.output_bytes, task_id});
    }

    uint64_t cursor_sum = 0U;
    for (uint32_t shard = 0U; shard < 8U; ++shard) {
        auto &shard_intervals = intervals[shard];
        std::sort(
            shard_intervals.begin(), shard_intervals.end(),
            [](const HeapInterval &left, const HeapInterval &right) {
                return left.begin < right.begin;
            }
        );
        uint64_t next = static_cast<uint64_t>(shard) * (UINT64_C(32) << 20U);
        for (const HeapInterval &interval : shard_intervals) {
            if (!RecordCheck(
                    interval.begin == next, failure, "heap-shard-gap-or-overlap", interval.task_id, UINT32_MAX, shard,
                    next, interval.begin
                )) {
                return false;
            }
            next = interval.end;
        }
        const uint64_t expected_shard = ExpectedHeapShardBytes(batches, shard);
        const uint64_t actual_shard = next - static_cast<uint64_t>(shard) * (UINT64_C(32) << 20U);
        const int64_t raw_cursor = state.heap.shard_cursors[shard].value;
        if (!RecordCheck(
                actual_shard == expected_shard, failure, "heap-shard-interval-total", UINT32_MAX, UINT32_MAX, shard,
                expected_shard, actual_shard
            ) ||
            !RecordCheck(
                raw_cursor >= 0 && static_cast<uint64_t>(raw_cursor) == expected_shard, failure, "heap-shard-cursor",
                UINT32_MAX, UINT32_MAX, shard, expected_shard, static_cast<uint64_t>(raw_cursor)
            ) ||
            !RecordCheck(
                AtomicPaddingMatches(state.heap.shard_cursors[shard], initial.heap.shard_cursors[shard]), failure,
                "heap-shard-padding-mutated", UINT32_MAX, UINT32_MAX, shard
            )) {
            return false;
        }
        cursor_sum += actual_shard;
    }
    std::sort(
        aggregate_intervals.begin(), aggregate_intervals.end(),
        [](const HeapInterval &left, const HeapInterval &right) {
            return left.begin < right.begin;
        }
    );
    uint64_t aggregate_cursor = 0U;
    for (const HeapInterval &interval : aggregate_intervals) {
        if (!RecordCheck(
                interval.begin == aggregate_cursor, failure, "heap-aggregate-gap-or-overlap", interval.task_id,
                UINT32_MAX, UINT32_MAX, aggregate_cursor, interval.begin
            )) {
            return false;
        }
        aggregate_cursor = interval.end;
    }
    const int64_t aggregate = state.heap.aggregate_vend.value;
    return RecordCheck(
               cursor_sum == expected_total, failure, "heap-interval-aggregate", UINT32_MAX, UINT32_MAX, UINT32_MAX,
               expected_total, cursor_sum
           ) &&
           RecordCheck(
               aggregate_cursor == expected_total, failure, "heap-vend-prefix-aggregate", UINT32_MAX, UINT32_MAX,
               UINT32_MAX, expected_total, aggregate_cursor
           ) &&
           RecordCheck(
               aggregate >= 0 && static_cast<uint64_t>(aggregate) == expected_total, failure, "heap-aggregate-vend",
               UINT32_MAX, UINT32_MAX, UINT32_MAX, expected_total, static_cast<uint64_t>(aggregate)
           ) &&
           RecordCheck(
               AtomicPaddingMatches(state.heap.aggregate_vend, initial.heap.aggregate_vend), failure,
               "heap-aggregate-padding-mutated"
           );
}

bool ValidateTask(
    const FullPaState &state, const FullPaState &initial, uint64_t nonce, uint32_t batches, uint32_t builder_count,
    const std::vector<uint64_t> &task_bases, uint32_t task_id, ValidationFailure *failure
) {
    const HostTaskOracle expected = BuildHostTaskOracle(task_id);
    const g0::FullPaTask &task = state.tasks[task_id];
    const g0::FullPaTask &initial_task = initial.tasks[task_id];
    const uint64_t vend = static_cast<uint64_t>(task.allocation.completion_vend_plus_one.value - 1);
    const uint8_t expected_meta = static_cast<uint8_t>(
        0x80U | (task_id + 1U == batches * kHostTasksPerBatch ? 0x40U : 0U) | static_cast<uint32_t>(expected.kind)
    );
    const uint8_t expected_route =
        expected.engine == HostEngine::None ? 1U : (expected.engine == HostEngine::Aic ? 7U : 11U);
    const g0::FullPaTaskPlan &plan = task.plan;
    const uint32_t actual_builder_owner = plan.builder_owner;
    if (!RecordCheck(
            g0::IsBuilderOwner(actual_builder_owner, builder_count), failure, "task-plan-builder-owner", task_id,
            actual_builder_owner, UINT32_MAX, g0::kBuilderOwner, actual_builder_owner
        )) {
        return false;
    }
    const uint32_t expected_builder_thread = g0::BuilderThreadForTask(task_id, builder_count);
    const uint32_t expected_builder_warp = expected_builder_thread / kHostWarpThreads;
    const uint32_t expected_builder_owner = g0::BuilderOwnerForThread(expected_builder_thread);
    if (!RecordCheck(
            plan.task_id == task_id, failure, "task-plan-id", task_id, UINT32_MAX, UINT32_MAX, task_id, plan.task_id
        ) ||
        !RecordCheck(
            actual_builder_owner == expected_builder_owner, failure, "task-plan-builder-owner-mapping", task_id,
            actual_builder_owner, UINT32_MAX, expected_builder_owner, actual_builder_owner
        ) ||
        !RecordCheck(
            plan.batch == expected.batch, failure, "task-plan-batch", task_id, UINT32_MAX, UINT32_MAX, expected.batch,
            plan.batch
        ) ||
        !RecordCheck(
            static_cast<uint32_t>(plan.kind) == static_cast<uint32_t>(expected.kind), failure, "task-plan-kind",
            task_id, UINT32_MAX, UINT32_MAX, static_cast<uint32_t>(expected.kind), static_cast<uint32_t>(plan.kind)
        ) ||
        !RecordCheck(
            static_cast<uint32_t>(plan.engine_class) == static_cast<uint32_t>(expected.engine), failure,
            "task-plan-engine", task_id, UINT32_MAX, UINT32_MAX, static_cast<uint32_t>(expected.engine),
            static_cast<uint32_t>(plan.engine_class)
        ) ||
        !RecordCheck(
            plan.output_count == expected.output_count, failure, "task-plan-output-count", task_id, UINT32_MAX,
            UINT32_MAX, expected.output_count, plan.output_count
        ) ||
        !RecordCheck(
            plan.payload_lines == expected.payload_lines, failure, "task-plan-payload-lines", task_id, UINT32_MAX,
            UINT32_MAX, expected.payload_lines, plan.payload_lines
        ) ||
        !RecordCheck(
            plan.builder_thread == expected_builder_thread, failure, "task-plan-builder-thread", task_id,
            actual_builder_owner, UINT32_MAX, expected_builder_thread, plan.builder_thread
        ) ||
        !RecordCheck(
            plan.builder_warp == expected_builder_warp, failure, "task-plan-builder-warp", task_id,
            actual_builder_owner, UINT32_MAX, expected_builder_warp, plan.builder_warp
        ) ||
        !RecordCheck(
            plan.encoded_meta == expected_meta, failure, "task-plan-encoded-meta", task_id, UINT32_MAX, UINT32_MAX,
            expected_meta, plan.encoded_meta
        ) ||
        !RecordCheck(
            plan.exec_route == expected_route, failure, "task-plan-exec-route", task_id, UINT32_MAX, UINT32_MAX,
            expected_route, plan.exec_route
        ) ||
        !RecordCheck(
            plan.reserved_bytes == expected.output_bytes, failure, "task-plan-reserved-bytes", task_id, UINT32_MAX,
            UINT32_MAX, expected.output_bytes, plan.reserved_bytes
        ) ||
        !RecordCheck(
            plan.launch_nonce == nonce, failure, "task-plan-nonce", task_id, UINT32_MAX, UINT32_MAX, nonce,
            plan.launch_nonce
        ) ||
        !RecordCheck(
            plan.metadata_insert_contract == HostMetadataInsertContract(task_id), failure,
            "task-plan-metadata-insert-contract", task_id, UINT32_MAX, UINT32_MAX,
            HostMetadataInsertContract(task_id), plan.metadata_insert_contract
        )) {
        return false;
    }

    if (!RecordCheck(
            task.completion.flag == 1, failure, "completion-flag", task_id, UINT32_MAX, UINT32_MAX, 1U,
            static_cast<uint64_t>(task.completion.flag)
        ) ||
        !RecordCheck(
            task.completion.vend == vend, failure, "completion-vend", task_id, UINT32_MAX, UINT32_MAX, vend,
            task.completion.vend
        ) ||
        !RecordCheck(
            task.completion.deps_prepared == static_cast<int64_t>(task_id) - 1, failure, "deps-prepared-canary",
            task_id, UINT32_MAX, UINT32_MAX, static_cast<uint64_t>(static_cast<int64_t>(task_id) - 1),
            static_cast<uint64_t>(task.completion.deps_prepared)
        ) ||
        !RecordCheck(
            std::memcmp(task.completion.padding, initial_task.completion.padding, sizeof(task.completion.padding)) == 0,
            failure, "completion-padding-mutated", task_id
        ) ||
        !RecordCheck(
            static_cast<int32_t>(task.insert_completion.value) ==
                (HostMetadataWriterIntentCount(task_id) != 0U ? static_cast<int32_t>(task_id) :
                                                                static_cast<int32_t>(task_id) - 1),
            failure, "insert-completion", task_id, UINT32_MAX, UINT32_MAX,
            HostMetadataWriterIntentCount(task_id) != 0U ?
                task_id : static_cast<uint64_t>(static_cast<int32_t>(task_id) - 1),
            static_cast<uint32_t>(static_cast<int32_t>(task.insert_completion.value))
        )) {
        return false;
    }

    for (uint32_t slot = 0U; slot < g0::kOutputsPerTask; ++slot) {
        const bool active = slot < expected.output_count;
        const int64_t expected_published = active ? static_cast<int64_t>(task_id) : -1;
        const int64_t expected_writer =
            active ? HostLatestMetadataWriterBefore(batches * kHostTasksPerBatch, task_id, slot) : -1;
        const uint64_t *published_words = reinterpret_cast<const uint64_t *>(&task.outputs.published[0]);
        const uint64_t *last_writer_words = reinterpret_cast<const uint64_t *>(&task.outputs.last_writer[0]);
        if (!RecordCheck(
                static_cast<int64_t>(published_words[slot]) == expected_published, failure, "output-published", task_id,
                UINT32_MAX, slot, static_cast<uint64_t>(expected_published),
                published_words[slot]
            ) ||
            !RecordCheck(
                static_cast<int64_t>(last_writer_words[slot]) == expected_writer, failure, "output-last-writer", task_id,
                UINT32_MAX, slot, static_cast<uint64_t>(expected_writer),
                last_writer_words[slot]
            )) {
            return false;
        }
        if (!active) {
            if (!RecordCheck(
                    std::memcmp(
                        &task.outputs.tensors[slot], &initial_task.outputs.tensors[slot],
                        sizeof(task.outputs.tensors[slot])
                    ) == 0,
                    failure, "inactive-output-descriptor-mutated", task_id, UINT32_MAX, slot
                )) {
                return false;
            }
            continue;
        }
        const HostTensorDesc expected_tensor = MakeOutputTensor(expected, slot, task_bases[task_id]);
        if (!RecordCheck(
                std::memcmp(&task.outputs.tensors[slot], &expected_tensor, sizeof(expected_tensor)) == 0, failure,
                "output-descriptor", task_id, UINT32_MAX, slot
            )) {
            return false;
        }
    }

    const uint32_t writer_count = HostMetadataWriterIntentCount(task_id);
    if (writer_count == 0U) {
        if (!RecordCheck(
                std::memcmp(&task.writer_history, &initial_task.writer_history, sizeof(task.writer_history)) == 0,
                failure, "unexpected-writer-history", task_id
            )) {
            return false;
        }
    } else {
        const g0::WriterHistoryCell &history = task.writer_history;
        if (!RecordCheck(
                history.magic == UINT32_C(0x57484953), failure, "history-magic", task_id, UINT32_MAX, UINT32_MAX,
                UINT32_C(0x57484953), history.magic
            ) ||
            !RecordCheck(
                history.writer_task == static_cast<int32_t>(task_id), failure, "history-writer", task_id, UINT32_MAX,
                UINT32_MAX, task_id, static_cast<uint64_t>(history.writer_task)
            ) ||
            !RecordCheck(
                history.count == writer_count, failure, "history-count", task_id, UINT32_MAX, UINT32_MAX,
                writer_count, history.count
            ) ||
            !RecordCheck(
                history.reserved == 0U, failure, "history-reserved", task_id, UINT32_MAX, UINT32_MAX, 0U,
                history.reserved
            )) {
            return false;
        }
        for (uint32_t index = 0U; index < writer_count; ++index) {
            HostSharedWriterIntent intent{};
            if (!HostMetadataWriterIntentAt(task_id, index, &intent)) {
                return false;
            }
            const uint32_t expected_key = intent.producer_task * 8U + intent.output_slot + 1U;
            const int32_t expected_previous =
                HostLatestMetadataWriterBefore(task_id, intent.producer_task, intent.output_slot);
            if (!RecordCheck(
                    history.entries[index].symbol_key == expected_key, failure, "history-symbol-key", task_id,
                    UINT32_MAX, index, expected_key, history.entries[index].symbol_key
                ) ||
                !RecordCheck(
                    history.entries[index].previous_writer == expected_previous, failure,
                    "history-previous-writer", task_id, UINT32_MAX, index,
                    static_cast<uint64_t>(expected_previous),
                    static_cast<uint64_t>(history.entries[index].previous_writer)
                )) {
                return false;
            }
        }
        const size_t written_history_bytes =
            offsetof(g0::WriterHistoryCell, entries) + writer_count * sizeof(g0::WriterHistoryRecord);
        const auto *history_bytes = reinterpret_cast<const uint8_t *>(&history);
        const auto *initial_bytes = reinterpret_cast<const uint8_t *>(&initial_task.writer_history);
        if (!RecordCheck(
                std::memcmp(
                    history_bytes + written_history_bytes, initial_bytes + written_history_bytes,
                    sizeof(history) - written_history_bytes
                ) == 0,
                failure, "history-tail-mutated", task_id
            )) {
            return false;
        }
    }

    const g0::FullPaBuildReport &report = task.build_report;
    const uint32_t metadata_predecessor = HostPreviousMetadataWriterTask(task_id);
    const uint32_t expected_phase_bits = expected.engine == HostEngine::None ? 0x17U : 0x0FU;
    if (!RecordCheck(
            report.task_id == task_id, failure, "build-report-task", task_id, UINT32_MAX, UINT32_MAX, task_id,
            report.task_id
        ) ||
        !RecordCheck(
            report.builder_thread == expected_builder_thread, failure, "build-report-thread", task_id,
            actual_builder_owner, UINT32_MAX, expected_builder_thread, report.builder_thread
        ) ||
        !RecordCheck(
            report.builder_warp == expected_builder_warp, failure, "build-report-warp", task_id, actual_builder_owner,
            UINT32_MAX, expected_builder_warp, report.builder_warp
        ) ||
        !RecordCheck(
            report.builder_lane == 0U, failure, "build-report-lane", task_id, UINT32_MAX, UINT32_MAX, 0U,
            report.builder_lane
        ) ||
        !RecordCheck(
            report.phase_bits == expected_phase_bits, failure, "build-report-phase-bits", task_id, UINT32_MAX,
            UINT32_MAX, expected_phase_bits, report.phase_bits
        ) ||
        !RecordCheck(
            report.output_count == expected.output_count, failure, "build-report-output-count", task_id, UINT32_MAX,
            UINT32_MAX, expected.output_count, report.output_count
        ) ||
        !RecordCheck(
            report.payload_words == expected.written_words, failure, "build-report-payload-words", task_id, UINT32_MAX,
            UINT32_MAX, expected.written_words, report.payload_words
        ) ||
        !RecordCheck(
            metadata_predecessor != UINT32_MAX ? report.insert_poll_count >= 1U :
                                                 report.insert_poll_count == 0U,
            failure, "build-report-insert-poll", task_id
        ) ||
        !RecordCheck(
            report.predecessor_observed ==
                (metadata_predecessor != UINT32_MAX ? static_cast<int64_t>(metadata_predecessor) : -1),
            failure, "build-report-predecessor", task_id, UINT32_MAX, UINT32_MAX,
            static_cast<uint64_t>(metadata_predecessor != UINT32_MAX ?
                                      static_cast<int64_t>(metadata_predecessor) : -1),
            static_cast<uint64_t>(report.predecessor_observed)
        ) ||
        !RecordCheck(
            report.prepare_count == 1U, failure, "build-report-prepare-count", task_id, UINT32_MAX, UINT32_MAX, 1U,
            report.prepare_count
        ) ||
        !RecordCheck(
            report.commit_count == 1U, failure, "build-report-commit-count", task_id, UINT32_MAX, UINT32_MAX, 1U,
            report.commit_count
        ) ||
        !RecordCheck(
            report.build_attempt_count == 1U, failure, "build-report-attempt-count", task_id,
            actual_builder_owner, UINT32_MAX, 1U, report.build_attempt_count
        ) ||
        !RecordCheck(
            report.build_win_count == 1U, failure, "build-report-win-count", task_id, actual_builder_owner, UINT32_MAX,
            1U, report.build_win_count
        ) ||
        !RecordCheck(
            report.launch_nonce == nonce, failure, "build-report-nonce", task_id, UINT32_MAX, UINT32_MAX, nonce,
            report.launch_nonce
        )) {
        return false;
    }

    if (expected.engine == HostEngine::None) {
        return RecordCheck(task.exec.control.state == 0, failure, "alloc-exec-control", task_id) &&
               RecordCheck(
                   std::memcmp(&task.exec.payload, &initial_task.exec.payload, sizeof(task.exec.payload)) == 0, failure,
                   "alloc-exec-payload-mutated", task_id
               ) &&
               RecordCheck(
                   std::memcmp(
                       &task.execution_witness, &initial_task.execution_witness, sizeof(task.execution_witness)
                   ) == 0,
                   failure, "alloc-witness-mutated", task_id
               );
    }

    const HostDecodedExecState decoded = DecodeHostExecState(task.exec.control.state);
    if (!RecordCheck(
            decoded.known_bits && decoded.phase == 4U && decoded.build_owner == actual_builder_owner &&
                decoded.engine == static_cast<uint32_t>(expected.engine) &&
                decoded.payload_lines == expected.payload_lines && decoded.task_id == task_id,
            failure, "exec-done-control", task_id
        ) ||
        !RecordCheck(
            (expected.engine == HostEngine::Aic && decoded.execute_owner < 32U) ||
                (expected.engine == HostEngine::Aiv && decoded.execute_owner >= g0::kBuilderOwner + builder_count &&
                 decoded.execute_owner < 96U),
            failure, "exec-owner-route", task_id, decoded.execute_owner
        ) ||
        !RecordCheck(
            std::memcmp(
                task.exec.control.padding, initial_task.exec.control.padding, sizeof(task.exec.control.padding)
            ) == 0,
            failure, "exec-control-padding-mutated", task_id
        )) {
        return false;
    }

    const uint64_t expected_header[8] = {
        task_id,
        0U,
        vend,
        static_cast<uint64_t>(static_cast<uint32_t>(expected.kind) - 1U) |
            (static_cast<uint64_t>(expected.payload_bytes) << 32U),
        static_cast<uint64_t>(expected.tensor_count) | (static_cast<uint64_t>(expected.scalar_count) << 16U) |
            (static_cast<uint64_t>(expected.fanin_count) << 32U) | (static_cast<uint64_t>(expected.engine) << 48U),
        UINT64_C(1) << 48U,
        0U,
        0U,
    };
    for (uint32_t word = 0U; word < 8U; ++word) {
        if (!RecordCheck(
                task.exec.payload.words[word] == expected_header[word], failure, "payload-header", task_id, UINT32_MAX,
                word, expected_header[word], task.exec.payload.words[word]
            )) {
            return false;
        }
    }
    for (uint32_t tensor_index = 0U; tensor_index < expected.tensor_count; ++tensor_index) {
        HostTensorDesc expected_tensor{};
        if (!RecordCheck(
                ExpectedPayloadTensor(expected, tensor_index, batches, task_bases, &expected_tensor), failure,
                "payload-tensor-oracle", task_id, UINT32_MAX, tensor_index
            )) {
            return false;
        }
        const HostTensorDesc actual_tensor = LoadPayloadTensor(task.exec, tensor_index);
        if (!RecordCheck(
                std::memcmp(&actual_tensor, &expected_tensor, sizeof(expected_tensor)) == 0, failure, "payload-tensor",
                task_id, UINT32_MAX, tensor_index
            )) {
            return false;
        }
    }
    const uint32_t scalar_offset = 8U + expected.tensor_count * 16U;
    for (uint32_t scalar = 0U; scalar < expected.scalar_count; ++scalar) {
        if (!RecordCheck(
                task.exec.payload.words[scalar_offset + scalar] == expected.scalars[scalar], failure, "payload-scalar",
                task_id, UINT32_MAX, scalar, expected.scalars[scalar], task.exec.payload.words[scalar_offset + scalar]
            )) {
            return false;
        }
    }
    const uint32_t fanin_offset = scalar_offset + expected.scalar_count;
    for (uint32_t edge = 0U; edge < expected.fanin_count; edge += 2U) {
        const uint64_t low = static_cast<uint32_t>(expected.predecessors[edge]);
        const uint64_t high =
            edge + 1U < expected.fanin_count ? static_cast<uint32_t>(expected.predecessors[edge + 1U]) : 0U;
        const uint64_t packed = low | (high << 32U);
        if (!RecordCheck(
                task.exec.payload.words[fanin_offset + edge / 2U] == packed, failure, "payload-fanin", task_id,
                UINT32_MAX, edge, packed, task.exec.payload.words[fanin_offset + edge / 2U]
            )) {
            return false;
        }
    }
    for (uint32_t word = expected.written_words; word < g0::kMaxPayloadWords; ++word) {
        if (!RecordCheck(
                task.exec.payload.words[word] == initial_task.exec.payload.words[word], failure, "payload-tail-mutated",
                task_id, UINT32_MAX, word, initial_task.exec.payload.words[word], task.exec.payload.words[word]
            )) {
            return false;
        }
    }

    const g0::FullPaExecutionWitness &witness = task.execution_witness;
    const uint64_t expected_witness_state =
        HostExecutionWitnessState(nonce, task_id, expected.kind, decoded.execute_owner);
    return RecordCheck(
               static_cast<uint64_t>(witness.state) == expected_witness_state, failure, "witness-state", task_id,
               decoded.execute_owner, UINT32_MAX, expected_witness_state, static_cast<uint64_t>(witness.state)
           ) &&
           RecordCheck(
               witness.launch_nonce == nonce, failure, "witness-nonce", task_id, decoded.execute_owner, UINT32_MAX,
               nonce, witness.launch_nonce
           ) &&
           RecordCheck(
               witness.witness_magic == UINT64_C(0x5749544E45535330), failure, "witness-magic", task_id,
               decoded.execute_owner, UINT32_MAX, UINT64_C(0x5749544E45535330), witness.witness_magic
           ) &&
           RecordCheck(
               witness.task_id == task_id, failure, "witness-task", task_id, decoded.execute_owner, UINT32_MAX, task_id,
               witness.task_id
           ) &&
           RecordCheck(
               static_cast<uint32_t>(witness.kind) == static_cast<uint32_t>(expected.kind), failure, "witness-kind",
               task_id, decoded.execute_owner, UINT32_MAX, static_cast<uint32_t>(expected.kind),
               static_cast<uint32_t>(witness.kind)
           ) &&
           RecordCheck(
               witness.execute_owner == decoded.execute_owner, failure, "witness-owner", task_id, decoded.execute_owner,
               UINT32_MAX, decoded.execute_owner, witness.execute_owner
           ) &&
           RecordCheck(
               witness.execution_count == 1U, failure, "witness-execution-count", task_id, decoded.execute_owner,
               UINT32_MAX, 1U, witness.execution_count
           ) &&
           RecordCheck(
               witness.completion_sequence == UINT64_C(0x0102030405), failure, "witness-completion-sequence", task_id,
               decoded.execute_owner, UINT32_MAX, UINT64_C(0x0102030405), witness.completion_sequence
           ) &&
           RecordCheck(
               witness.output_checksum == ExpectedWitnessChecksum(expected.kind), failure, "witness-output-checksum",
               task_id, decoded.execute_owner, UINT32_MAX, ExpectedWitnessChecksum(expected.kind),
               witness.output_checksum
           ) &&
           RecordCheck(
               witness.fanin_ready_prefix == expected.fanin_count, failure, "witness-fanin-ready-prefix", task_id,
               decoded.execute_owner, UINT32_MAX, expected.fanin_count, witness.fanin_ready_prefix
           );
}

bool ValidateBuilderThreads(
    const FullPaState &state, const FullPaState &initial, uint64_t nonce, uint32_t task_count, uint32_t builder_count,
    ValidationFailure *failure
) {
    const uint32_t active_thread_count = g0::BuilderThreadCount(builder_count);
    std::vector<uint32_t> expected_wins(active_thread_count, 0U);
    std::vector<uint32_t> expected_first(active_thread_count, UINT32_MAX);
    std::vector<uint32_t> expected_last(active_thread_count, UINT32_MAX);
    std::vector<uint32_t> expected_waits(active_thread_count, 0U);
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const uint32_t thread_id = state.tasks[task_id].build_report.builder_thread;
        if (!RecordCheck(
                thread_id < active_thread_count && g0::BuilderThreadActive(thread_id, builder_count), failure,
                "builder-winner-thread-range", task_id, UINT32_MAX, thread_id, active_thread_count - 1U, thread_id
            )) {
            return false;
        }
        ++expected_wins[thread_id];
        if (expected_first[thread_id] == UINT32_MAX) {
            expected_first[thread_id] = task_id;
        }
        expected_last[thread_id] = task_id;
        expected_waits[thread_id] += HostPreviousMetadataWriterTask(task_id) == UINT32_MAX ? 0U : 1U;
    }

    uint64_t win_sum = 0U;
    uint64_t attempt_sum = 0U;
    uint64_t prepare_sum = 0U;
    uint64_t commit_sum = 0U;
    uint64_t loss_sum = 0U;
    for (uint32_t thread_id = 0U; thread_id < active_thread_count; ++thread_id) {
        const g0::FullPaBuilderThreadReport &report = state.builder_threads[thread_id];
        const uint32_t warp = thread_id / kHostWarpThreads;
        const uint32_t lane = thread_id % kHostWarpThreads;
        const bool leader = lane == 0U;
        if (!leader) {
            if (!RecordCheck(
                    std::memcmp(&report, &initial.builder_threads[thread_id], sizeof(report)) == 0, failure,
                    "inactive-builder-lane-mutated", UINT32_MAX, UINT32_MAX, thread_id
                )) {
                return false;
            }
            continue;
        }
        const uint32_t expected_attempts = g0::BuilderExpectedTaskCount(thread_id, task_count, builder_count);
        const uint32_t wins = expected_wins[thread_id];
        if (!RecordCheck(
                wins <= expected_attempts, failure, "builder-thread-win-exceeds-attempts", UINT32_MAX, UINT32_MAX,
                thread_id, expected_attempts, wins
            )) {
            return false;
        }
        const uint32_t claim_losses = 0U;
        const uint64_t expected_checksum = ExpectedBuilderChecksum(
            nonce, thread_id, task_count, wins, expected_first[thread_id], expected_last[thread_id], expected_attempts,
            wins, wins, expected_waits[thread_id], claim_losses
        );
        if (!RecordCheck(
                report.thread_id == thread_id, failure, "builder-thread-id", UINT32_MAX, UINT32_MAX, thread_id,
                thread_id, report.thread_id
            ) ||
            !RecordCheck(
                report.warp_id == warp, failure, "builder-thread-warp", UINT32_MAX, UINT32_MAX, thread_id, warp,
                report.warp_id
            ) ||
            !RecordCheck(
                report.lane_id == lane, failure, "builder-thread-lane", UINT32_MAX, UINT32_MAX, thread_id, lane,
                report.lane_id
            ) ||
            !RecordCheck(
                report.active_leader == (leader ? 1U : 0U), failure, "builder-thread-active", UINT32_MAX, UINT32_MAX,
                thread_id, leader ? 1U : 0U, report.active_leader
            ) ||
            !RecordCheck(
                report.task_count == wins, failure, "builder-thread-task-count", UINT32_MAX, UINT32_MAX, thread_id,
                wins, report.task_count
            ) ||
            !RecordCheck(
                report.first_task == expected_first[thread_id], failure, "builder-thread-first-task", UINT32_MAX,
                UINT32_MAX, thread_id, expected_first[thread_id], report.first_task
            ) ||
            !RecordCheck(
                report.last_task == expected_last[thread_id], failure, "builder-thread-last-task", UINT32_MAX,
                UINT32_MAX, thread_id, expected_last[thread_id], report.last_task
            ) ||
            !RecordCheck(
                report.task_state_access_count == expected_attempts, failure, "builder-thread-state-access", UINT32_MAX,
                UINT32_MAX, thread_id, expected_attempts, report.task_state_access_count
            ) ||
            !RecordCheck(
                report.prepare_count == wins, failure, "builder-thread-prepare", UINT32_MAX, UINT32_MAX, thread_id,
                wins, report.prepare_count
            ) ||
            !RecordCheck(
                report.commit_count == wins, failure, "builder-thread-commit", UINT32_MAX, UINT32_MAX, thread_id, wins,
                report.commit_count
            ) ||
            !RecordCheck(
                report.insert_wait_count == expected_waits[thread_id], failure, "builder-thread-waits", UINT32_MAX,
                UINT32_MAX, thread_id, expected_waits[thread_id], report.insert_wait_count
            ) ||
            !RecordCheck(
                report.claim_lost_count == claim_losses, failure, "builder-thread-claim-lost", UINT32_MAX, UINT32_MAX,
                thread_id, claim_losses, report.claim_lost_count
            ) ||
            !RecordCheck(
                report.launch_nonce == nonce, failure, "builder-thread-nonce", UINT32_MAX, UINT32_MAX, thread_id, nonce,
                report.launch_nonce
            ) ||
            !RecordCheck(
                report.checksum == expected_checksum, failure, "builder-thread-checksum", UINT32_MAX, UINT32_MAX,
                thread_id, expected_checksum, report.checksum
            )) {
            return false;
        }
        win_sum += report.task_count;
        attempt_sum += report.task_state_access_count;
        prepare_sum += report.prepare_count;
        commit_sum += report.commit_count;
        loss_sum += report.claim_lost_count;
    }
    for (uint32_t thread_id = active_thread_count; thread_id < g0::kMaxBuilderThreadCount; ++thread_id) {
        if (!RecordCheck(
                std::memcmp(
                    &state.builder_threads[thread_id], &initial.builder_threads[thread_id],
                    sizeof(state.builder_threads[thread_id])
                ) == 0,
                failure, "inactive-builder-instance-mutated", UINT32_MAX, UINT32_MAX, thread_id
            )) {
            return false;
        }
    }
    return RecordCheck(
               win_sum == task_count, failure, "builder-thread-win-sum", UINT32_MAX, UINT32_MAX, UINT32_MAX, task_count,
               win_sum
           ) &&
           RecordCheck(
               attempt_sum == task_count, failure, "builder-thread-attempt-sum",
               UINT32_MAX, UINT32_MAX, UINT32_MAX, task_count, attempt_sum
           ) &&
           RecordCheck(
               prepare_sum == task_count, failure, "builder-thread-prepare-sum", UINT32_MAX, UINT32_MAX, UINT32_MAX,
               task_count, prepare_sum
           ) &&
           RecordCheck(
               commit_sum == task_count, failure, "builder-thread-commit-sum", UINT32_MAX, UINT32_MAX, UINT32_MAX,
               task_count, commit_sum
           ) &&
           RecordCheck(
               loss_sum == 0U, failure, "builder-thread-claim-loss-sum", UINT32_MAX, UINT32_MAX, UINT32_MAX,
               0U, loss_sum
           );
}

bool ValidateResetTokenControl(
    const g0::ExecutionTokenControl &control, uint32_t owner, uint32_t slot, ValidationFailure *failure
) {
    return RecordCheck(
               control.phase == g0::ExecTokenPhase::Idle, failure, "token-control-phase-not-reset", UINT32_MAX, owner,
               slot, static_cast<uint32_t>(g0::ExecTokenPhase::Idle), static_cast<uint32_t>(control.phase)
           ) &&
           RecordCheck(
               control.task_id == UINT32_MAX, failure, "token-control-task-not-reset", UINT32_MAX, owner, slot,
               UINT32_MAX, control.task_id
           ) &&
           RecordCheck(
               control.build_owner == UINT32_MAX, failure, "token-control-builder-not-reset", UINT32_MAX, owner, slot,
               UINT32_MAX, control.build_owner
           ) &&
           RecordCheck(
               control.execute_owner == UINT32_MAX, failure, "token-control-executor-not-reset", UINT32_MAX, owner,
               slot, UINT32_MAX, control.execute_owner
           ) &&
           RecordCheck(
               control.engine_class == g0::ExecEngineClass::None, failure, "token-control-engine-not-reset", UINT32_MAX,
               owner, slot, static_cast<uint32_t>(g0::ExecEngineClass::None),
               static_cast<uint32_t>(control.engine_class)
           ) &&
           RecordCheck(
               control.payload_lines == 0U, failure, "token-control-lines-not-reset", UINT32_MAX, owner, slot, 0U,
               control.payload_lines
           ) &&
           RecordCheck(
               control.payload_bytes == 0U, failure, "token-control-bytes-not-reset", UINT32_MAX, owner, slot, 0U,
               control.payload_bytes
           ) &&
           RecordCheck(
               control.fanin_ready_prefix == 0U, failure, "token-control-fanin-not-reset", UINT32_MAX, owner, slot, 0U,
               control.fanin_ready_prefix
           ) &&
           RecordCheck(
               control.payload_address == 0U, failure, "token-control-payload-not-reset", UINT32_MAX, owner, slot, 0U,
               control.payload_address
           ) &&
           RecordCheck(
               control.completion_vend == 0U, failure, "token-control-vend-not-reset", UINT32_MAX, owner, slot, 0U,
               control.completion_vend
           ) &&
           RecordCheck(
               control.function_and_reference == 0U, failure, "token-control-function-not-reset", UINT32_MAX, owner,
               slot, 0U, control.function_and_reference
           ) &&
           RecordCheck(
               control.shape_and_scalar_offset == 0U, failure, "token-control-shape-not-reset", UINT32_MAX, owner, slot,
               0U, control.shape_and_scalar_offset
           );
}

bool ValidateRetainedTokenDispatch(
    const FullPaState &state, const FullPaState &initial, uint64_t device_state_address, uint32_t batches,
    uint32_t builder_count, uint32_t owner, uint32_t slot, std::vector<uint8_t> *retained_tasks,
    ValidationFailure *failure
) {
    const g0::ExecutionToken &token = state.tokens[owner][slot];
    const g0::ExecutionToken &initial_token = initial.tokens[owner][slot];
    const uint32_t used_tokens = std::min(state.roles[owner].ticket_count, g0::kTokensPerOwner);
    if (slot >= used_tokens) {
        return RecordCheck(
            std::memcmp(&token.dispatch, &initial_token.dispatch, sizeof(token.dispatch)) == 0, failure,
            g0::IsBuilderOwner(owner, builder_count) ? "builder-token-dispatch-mutated" :
                                                       "unused-token-dispatch-mutated",
            UINT32_MAX, owner, slot
        );
    }

    std::array<uint64_t, g0::kLocalContextBytes / sizeof(uint64_t)> local_context{};
    std::memcpy(local_context.data(), token.dispatch.local_context, sizeof(token.dispatch.local_context));
    const uint32_t task_count = batches * kHostTasksPerBatch;
    if (!RecordCheck(
            local_context[0] < task_count, failure, "token-retained-task-range", UINT32_MAX, owner, slot,
            task_count - 1U, local_context[0]
        )) {
        return false;
    }
    const uint32_t task_id = static_cast<uint32_t>(local_context[0]);
    const HostTaskOracle task = BuildHostTaskOracle(task_id);
    const HostEngine owner_engine = owner < g0::kAicOwnerCount ? HostEngine::Aic : HostEngine::Aiv;
    if (!RecordCheck(
            task.engine == owner_engine, failure, "token-retained-engine", task_id, owner, slot,
            static_cast<uint32_t>(owner_engine), static_cast<uint32_t>(task.engine)
        ) ||
        !RecordCheck(
            state.tasks[task_id].execution_witness.execute_owner == owner, failure, "token-retained-witness-owner",
            task_id, owner, slot, owner, state.tasks[task_id].execution_witness.execute_owner
        ) ||
        !RecordCheck(
            (*retained_tasks)[task_id] == 0U, failure, "token-retained-task-duplicate", task_id, owner, slot, 0U,
            (*retained_tasks)[task_id]
        )) {
        return false;
    }
    (*retained_tasks)[task_id] = 1U;

    const std::array<uint64_t, g0::kLocalContextBytes / sizeof(uint64_t)> expected_local = {
        task_id, owner, task.payload_bytes, task.payload_lines, task.fanin_count, state.tasks[task_id].completion.vend,
    };
    for (uint32_t word = 0U; word < expected_local.size(); ++word) {
        if (!RecordCheck(
                local_context[word] == expected_local[word], failure, "token-retained-local-context", task_id, owner,
                slot * static_cast<uint32_t>(expected_local.size()) + word, expected_local[word], local_context[word]
            )) {
            return false;
        }
    }
    uint32_t global_batches = 0U;
    std::memcpy(&global_batches, token.dispatch.global_context, sizeof(global_batches));
    if (!RecordCheck(
            global_batches == batches, failure, "token-retained-global-context", task_id, owner, slot, batches,
            global_batches
        )) {
        return false;
    }

    const uint64_t token_address =
        device_state_address + offsetof(FullPaState, tokens) +
        (static_cast<uint64_t>(owner) * g0::kTokensPerOwner + slot) * sizeof(g0::ExecutionToken);
    const uint64_t expected_local_address =
        token_address + offsetof(g0::ExecutionToken, dispatch) + offsetof(g0::ExecutionDispatchBinding, local_context);
    const uint64_t expected_global_address =
        token_address + offsetof(g0::ExecutionToken, dispatch) + offsetof(g0::ExecutionDispatchBinding, global_context);
    if (!RecordCheck(
            token.dispatch.args[g0::kDispatchLocalContextIndex] == expected_local_address, failure,
            "token-retained-local-pointer", task_id, owner, slot, expected_local_address,
            token.dispatch.args[g0::kDispatchLocalContextIndex]
        ) ||
        !RecordCheck(
            token.dispatch.args[g0::kDispatchGlobalContextIndex] == expected_global_address, failure,
            "token-retained-global-pointer", task_id, owner, slot, expected_global_address,
            token.dispatch.args[g0::kDispatchGlobalContextIndex]
        )) {
        return false;
    }

    const uint64_t payload_address = device_state_address + offsetof(FullPaState, tasks) +
                                     static_cast<uint64_t>(task_id) * sizeof(g0::FullPaTask) +
                                     offsetof(g0::FullPaTask, exec) + offsetof(g0::SharedExecCell, payload);
    for (uint32_t tensor = 0U; tensor < task.tensor_count; ++tensor) {
        const uint64_t expected =
            payload_address +
            static_cast<uint64_t>(g0::kPayloadHeaderWords + tensor * g0::kTensorDescWords) * sizeof(uint64_t);
        if (!RecordCheck(
                token.dispatch.args[tensor] == expected, failure, "token-retained-tensor-arg", task_id, owner,
                slot * g0::kDispatchArgCount + tensor, expected, token.dispatch.args[tensor]
            )) {
            return false;
        }
    }
    for (uint32_t scalar = 0U; scalar < task.scalar_count; ++scalar) {
        const uint32_t arg = task.tensor_count + scalar;
        if (!RecordCheck(
                token.dispatch.args[arg] == task.scalars[scalar], failure, "token-retained-scalar-arg", task_id, owner,
                slot * g0::kDispatchArgCount + arg, task.scalars[scalar], token.dispatch.args[arg]
            )) {
            return false;
        }
    }
    for (uint32_t arg = 9U; arg < g0::kDispatchLocalContextIndex; ++arg) {
        if (!RecordCheck(
                token.dispatch.args[arg] == initial_token.dispatch.args[arg], failure, "token-unused-arg-mutated",
                task_id, owner, slot * g0::kDispatchArgCount + arg, initial_token.dispatch.args[arg],
                token.dispatch.args[arg]
            )) {
            return false;
        }
    }
    return RecordCheck(
        std::memcmp(token.dispatch.padding, initial_token.dispatch.padding, sizeof(token.dispatch.padding)) == 0,
        failure, "token-dispatch-padding-mutated", task_id, owner, slot
    );
}

bool ValidateDispatchAndTokens(
    const FullPaState &state, const FullPaState &initial, uint64_t device_state_address, uint32_t batches,
    uint32_t builder_count, ValidationFailure *failure
) {
    const uint32_t engine_tasks = batches * 2U;
    const uint64_t expected_aic_next = static_cast<uint64_t>(engine_tasks) + 32U;
    const uint64_t expected_aiv_next = static_cast<uint64_t>(engine_tasks) + g0::AivExecutorCount(builder_count);
    if (!RecordCheck(
            state.exec_dispatch.aic_next.value >= 0 &&
                static_cast<uint64_t>(state.exec_dispatch.aic_next.value) == expected_aic_next,
            failure, "aic-dispatch-cursor", UINT32_MAX, UINT32_MAX, UINT32_MAX, expected_aic_next,
            static_cast<uint64_t>(state.exec_dispatch.aic_next.value)
        ) ||
        !RecordCheck(
            state.exec_dispatch.aiv_next.value >= 0 &&
                static_cast<uint64_t>(state.exec_dispatch.aiv_next.value) == expected_aiv_next,
            failure, "aiv-dispatch-cursor", UINT32_MAX, UINT32_MAX, UINT32_MAX, expected_aiv_next,
            static_cast<uint64_t>(state.exec_dispatch.aiv_next.value)
        ) ||
        !RecordCheck(
            AtomicPaddingMatches(state.exec_dispatch.aic_next, initial.exec_dispatch.aic_next), failure,
            "aic-dispatch-padding-mutated"
        ) ||
        !RecordCheck(
            AtomicPaddingMatches(state.exec_dispatch.aiv_next, initial.exec_dispatch.aiv_next), failure,
            "aiv-dispatch-padding-mutated"
        ) ||
        !RecordCheck(
            state.exec_dispatch.aic_task_count == engine_tasks, failure, "aic-dispatch-count", UINT32_MAX, UINT32_MAX,
            UINT32_MAX, engine_tasks, state.exec_dispatch.aic_task_count
        ) ||
        !RecordCheck(
            state.exec_dispatch.aiv_task_count == engine_tasks, failure, "aiv-dispatch-count", UINT32_MAX, UINT32_MAX,
            UINT32_MAX, engine_tasks, state.exec_dispatch.aiv_task_count
        ) ||
        !RecordCheck(
            std::memcmp(
                state.exec_dispatch.header_padding, initial.exec_dispatch.header_padding,
                sizeof(state.exec_dispatch.header_padding)
            ) == 0,
            failure, "dispatch-header-padding-mutated"
        )) {
        return false;
    }
    for (uint32_t index = 0U; index < engine_tasks; ++index) {
        const uint32_t expected_aic = ExpectedExecutionTaskId(HostEngine::Aic, index);
        const uint32_t expected_aiv = ExpectedExecutionTaskId(HostEngine::Aiv, index);
        if (!RecordCheck(
                state.exec_dispatch.aic_task_ids[index] == expected_aic, failure, "aic-dispatch-id", expected_aic,
                UINT32_MAX, index, expected_aic, state.exec_dispatch.aic_task_ids[index]
            ) ||
            !RecordCheck(
                state.exec_dispatch.aiv_task_ids[index] == expected_aiv, failure, "aiv-dispatch-id", expected_aiv,
                UINT32_MAX, index, expected_aiv, state.exec_dispatch.aiv_task_ids[index]
            )) {
            return false;
        }
    }
    if (!RecordCheck(
            std::memcmp(
                state.exec_dispatch.aic_task_ids, initial.exec_dispatch.aic_task_ids,
                sizeof(state.exec_dispatch.aic_task_ids)
            ) == 0,
            failure, "aic-dispatch-plan-mutated"
        ) ||
        !RecordCheck(
            std::memcmp(
                state.exec_dispatch.aiv_task_ids, initial.exec_dispatch.aiv_task_ids,
                sizeof(state.exec_dispatch.aiv_task_ids)
            ) == 0,
            failure, "aiv-dispatch-plan-mutated"
        )) {
        return false;
    }

    std::vector<uint8_t> retained_tasks(batches * kHostTasksPerBatch, 0U);
    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        for (uint32_t slot = 0U; slot < g0::kTokensPerOwner; ++slot) {
            const g0::ExecutionToken &token = state.tokens[owner][slot];
            if (!ValidateResetTokenControl(token.control, owner, slot, failure) ||
                !ValidateRetainedTokenDispatch(
                    state, initial, device_state_address, batches, builder_count, owner, slot, &retained_tasks, failure
                )) {
                return false;
            }
        }
    }
    return true;
}

bool ValidateRolesAndDrain(
    const FullPaState &state, const FullPaState &initial, uint64_t nonce, uint32_t batches, uint32_t builder_count,
    ValidationFailure *failure
) {
    uint64_t task_kinds[5] = {};
    uint64_t total_execute = 0U;
    uint64_t total_tickets = 0U;
    uint64_t total_exhausted = 0U;
    uint64_t group_completions[16] = {};
    uint32_t group_arrivers[16] = {};
    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        const g0::FullPaRoleResult &role = state.roles[owner];
        const uint32_t expected_role = static_cast<uint32_t>(g0::OwnerRoleAt(owner, builder_count));
        const uint32_t expected_build_commit = 0U;
        const uint32_t physical = owner < 32U ? owner : (owner - 32U) / 2U;
        const uint32_t group = physical % 16U;
        uint64_t kind_sum = 0U;
        for (uint32_t kind = 0U; kind < 5U; ++kind) {
            const uint32_t count = role.completed_by_kind[kind];
            const bool legal_kind =
                owner < g0::kAicOwnerCount ?
                    (kind == 1U || kind == 3U) :
                    (g0::OwnerCanExecute(owner, g0::ExecEngineClass::Aiv, builder_count) && (kind == 2U || kind == 4U));
            if (!RecordCheck(
                    legal_kind || count == 0U, failure, "role-illegal-kind-count", UINT32_MAX, owner, kind, 0U, count
                )) {
                return false;
            }
            kind_sum += count;
            task_kinds[kind] += count;
        }
        const bool has_ticket = role.ticket_count != 0U;
        if (!RecordCheck(
                role.owner == owner, failure, "role-owner", UINT32_MAX, owner, UINT32_MAX, owner, role.owner
            ) ||
            !RecordCheck(
                static_cast<uint32_t>(role.role) == expected_role, failure, "role-type", UINT32_MAX, owner, UINT32_MAX,
                expected_role, static_cast<uint32_t>(role.role)
            ) ||
            !RecordCheck(
                role.physical_block == physical, failure, "role-physical-block", UINT32_MAX, owner, UINT32_MAX,
                physical, role.physical_block
            ) ||
            !RecordCheck(
                role.drain_group == group, failure, "role-drain-group", UINT32_MAX, owner, UINT32_MAX, group,
                role.drain_group
            ) ||
            !RecordCheck(
                role.build_count == expected_build_commit, failure, "role-simt-build-summary", UINT32_MAX, owner,
                UINT32_MAX, expected_build_commit, role.build_count
            ) ||
            !RecordCheck(
                role.commit_count == expected_build_commit, failure, "role-simt-commit-summary", UINT32_MAX, owner,
                UINT32_MAX, expected_build_commit, role.commit_count
            ) ||
            !RecordCheck(
                role.execute_count == kind_sum, failure, "role-execute-kind-sum", UINT32_MAX, owner, UINT32_MAX,
                kind_sum, role.execute_count
            ) ||
            !RecordCheck(
                role.ticket_count == role.execute_count, failure, "role-ticket-count", UINT32_MAX, owner, UINT32_MAX,
                role.execute_count, role.ticket_count
            ) ||
            !RecordCheck(
                role.exhausted_ticket_count == (g0::IsBuilderOwner(owner, builder_count) ? 0U : 1U), failure,
                "role-exhausted-ticket-count", UINT32_MAX, owner, UINT32_MAX,
                g0::IsBuilderOwner(owner, builder_count) ? 0U : 1U, role.exhausted_ticket_count
            ) ||
            !RecordCheck(
                role.claim_count == role.execute_count, failure, "role-claim-count", UINT32_MAX, owner, UINT32_MAX,
                role.execute_count, role.claim_count
            ) ||
            !RecordCheck(
                role.claim_lost_count == 0U, failure, "role-claim-lost", UINT32_MAX, owner, UINT32_MAX, 0U,
                role.claim_lost_count
            ) ||
            !RecordCheck(
                has_ticket ? (role.max_busy_tokens >= 1U && role.max_busy_tokens <= 4U) : role.max_busy_tokens == 0U,
                failure, "role-max-busy", UINT32_MAX, owner, UINT32_MAX, has_ticket ? 1U : 0U, role.max_busy_tokens
            ) ||
            !RecordCheck(
                role.final_busy_tokens == 0U, failure, "role-final-busy", UINT32_MAX, owner, UINT32_MAX, 0U,
                role.final_busy_tokens
            ) ||
            !RecordCheck(
                role.drain_arrival_count == 1U, failure, "role-drain-arrival", UINT32_MAX, owner, UINT32_MAX, 1U,
                role.drain_arrival_count
            ) ||
            !RecordCheck(
                role.fatal_count == 0U, failure, "role-fatal-count", UINT32_MAX, owner, UINT32_MAX, 0U, role.fatal_count
            ) ||
            !RecordCheck(
                role.launch_nonce == nonce, failure, "role-nonce", UINT32_MAX, owner, UINT32_MAX, nonce,
                role.launch_nonce
            )) {
            return false;
        }
        for (uint32_t reserved = 0U; reserved < sizeof(role.reserved) / sizeof(role.reserved[0]); ++reserved) {
            if (!RecordCheck(
                    role.reserved[reserved] == 0U, failure, "role-reserved", UINT32_MAX, owner, reserved, 0U,
                    role.reserved[reserved]
                )) {
                return false;
            }
        }
        total_execute += role.execute_count;
        total_tickets += role.ticket_count;
        total_exhausted += role.exhausted_ticket_count;
        group_completions[group] += role.execute_count;
        ++group_arrivers[group];
    }
    const uint64_t expected_kernel_tasks = static_cast<uint64_t>(batches) * 4U;
    if (!RecordCheck(
            task_kinds[0] == 0U && task_kinds[1] == batches && task_kinds[2] == batches && task_kinds[3] == batches &&
                task_kinds[4] == batches,
            failure, "role-kind-global-counts"
        ) ||
        !RecordCheck(
            total_execute == expected_kernel_tasks, failure, "role-execute-total", UINT32_MAX, UINT32_MAX, UINT32_MAX,
            expected_kernel_tasks, total_execute
        ) ||
        !RecordCheck(
            total_tickets == expected_kernel_tasks, failure, "role-ticket-total", UINT32_MAX, UINT32_MAX, UINT32_MAX,
            expected_kernel_tasks, total_tickets
        ) ||
        !RecordCheck(
            total_exhausted == g0::ExecutorCount(builder_count), failure, "role-exhausted-total", UINT32_MAX,
            UINT32_MAX, UINT32_MAX, g0::ExecutorCount(builder_count), total_exhausted
        )) {
        return false;
    }

    const auto drain_value = [&state](const g0::AtomicLine &line) {
        return line.value;
    };
    if (!RecordCheck(
            drain_value(state.drain.builder_started) == static_cast<int64_t>(builder_count), failure,
            "drain-builder-started", UINT32_MAX, UINT32_MAX, UINT32_MAX, builder_count,
            static_cast<uint64_t>(drain_value(state.drain.builder_started))
        ) ||
        !RecordCheck(
            drain_value(state.drain.builder_finished) == 1, failure, "drain-builder-finished", UINT32_MAX, UINT32_MAX,
            UINT32_MAX, 1U, static_cast<uint64_t>(drain_value(state.drain.builder_finished))
        ) ||
        !RecordCheck(
            drain_value(state.drain.done_count) == static_cast<int64_t>(expected_kernel_tasks), failure,
            "drain-done-count", UINT32_MAX, UINT32_MAX, UINT32_MAX, expected_kernel_tasks,
            static_cast<uint64_t>(drain_value(state.drain.done_count))
        ) ||
        !RecordCheck(
            drain_value(state.drain.alloc_done) == static_cast<int64_t>(batches), failure, "drain-alloc-done",
            UINT32_MAX, UINT32_MAX, UINT32_MAX, batches, static_cast<uint64_t>(drain_value(state.drain.alloc_done))
        ) ||
        !RecordCheck(
            drain_value(state.drain.aic_done) == static_cast<int64_t>(batches * 2U), failure, "drain-aic-done",
            UINT32_MAX, UINT32_MAX, UINT32_MAX, batches * 2U, static_cast<uint64_t>(drain_value(state.drain.aic_done))
        ) ||
        !RecordCheck(
            drain_value(state.drain.aiv_done) == static_cast<int64_t>(batches * 2U), failure, "drain-aiv-done",
            UINT32_MAX, UINT32_MAX, UINT32_MAX, batches * 2U, static_cast<uint64_t>(drain_value(state.drain.aiv_done))
        ) ||
        !RecordCheck(
            drain_value(state.drain.root_finished) == 1, failure, "drain-root-finished", UINT32_MAX, UINT32_MAX,
            UINT32_MAX, 1U, static_cast<uint64_t>(drain_value(state.drain.root_finished))
        )) {
        return false;
    }
    const g0::AtomicLine *drain_lines[] = {
        &state.drain.builder_started, &state.drain.builder_finished, &state.drain.done_count,
        &state.drain.alloc_done,      &state.drain.aic_done,         &state.drain.aiv_done,
        &state.drain.root_finished,
    };
    const g0::AtomicLine *initial_drain_lines[] = {
        &initial.drain.builder_started, &initial.drain.builder_finished, &initial.drain.done_count,
        &initial.drain.alloc_done,      &initial.drain.aic_done,         &initial.drain.aiv_done,
        &initial.drain.root_finished,
    };
    for (uint32_t line = 0U; line < sizeof(drain_lines) / sizeof(drain_lines[0]); ++line) {
        if (!RecordCheck(
                AtomicPaddingMatches(*drain_lines[line], *initial_drain_lines[line]), failure, "drain-padding-mutated",
                UINT32_MAX, UINT32_MAX, line
            )) {
            return false;
        }
    }
    for (uint32_t group = 0U; group < 16U; ++group) {
        const uint64_t expected_raw = (group_completions[group] << 8U) | 6U;
        if (!RecordCheck(
                group_arrivers[group] == 6U, failure, "drain-group-role-count", UINT32_MAX, UINT32_MAX, group, 6U,
                group_arrivers[group]
            ) ||
            !RecordCheck(
                state.drain.arrivals[group].value >= 0 &&
                    static_cast<uint64_t>(state.drain.arrivals[group].value) == expected_raw,
                failure, "drain-group-value", UINT32_MAX, UINT32_MAX, group, expected_raw,
                static_cast<uint64_t>(state.drain.arrivals[group].value)
            ) ||
            !RecordCheck(
                AtomicPaddingMatches(state.drain.arrivals[group], initial.drain.arrivals[group]), failure,
                "drain-group-padding-mutated", UINT32_MAX, UINT32_MAX, group
            )) {
            return false;
        }
    }
    return true;
}

uint32_t FloatBits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32 bits");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void InitializeWorkspace(std::vector<float> *workspace) {
    workspace->assign(static_cast<size_t>(kHostWorkloadTiles) * kHostWorkloadTileElements, kHostWorkloadSentinel);
    std::fill_n(workspace->data(), kHostWorkloadTileElements, kHostWorkloadInputA);
    std::fill_n(workspace->data() + kHostWorkloadTileElements, kHostWorkloadTileElements, kHostWorkloadInputB);
}

bool ValidateWorkspace(
    const FullPaState &state, const std::vector<float> &workspace, uint32_t builder_count, ValidationFailure *failure
) {
    if (!RecordCheck(
            workspace.size() == static_cast<size_t>(kHostWorkloadTiles) * kHostWorkloadTileElements, failure,
            "workspace-size", UINT32_MAX, UINT32_MAX, UINT32_MAX,
            static_cast<uint64_t>(kHostWorkloadTiles) * kHostWorkloadTileElements, workspace.size()
        )) {
        return false;
    }
    const auto validate_tile =
        [&workspace, failure](uint32_t tile, float expected, const char *reason, uint32_t owner, uint32_t slot) {
            const size_t begin = static_cast<size_t>(tile) * kHostWorkloadTileElements;
            const uint32_t expected_bits = FloatBits(expected);
            for (uint32_t element = 0U; element < kHostWorkloadTileElements; ++element) {
                const uint32_t actual_bits = FloatBits(workspace[begin + element]);
                if (!RecordCheck(
                        actual_bits == expected_bits, failure, reason, UINT32_MAX, owner,
                        slot * kHostWorkloadTileElements + element, expected_bits, actual_bits
                    )) {
                    return false;
                }
            }
            return true;
        };
    if (!validate_tile(0U, kHostWorkloadInputA, "workspace-input-a-mutated", UINT32_MAX, 0U) ||
        !validate_tile(1U, kHostWorkloadInputB, "workspace-input-b-mutated", UINT32_MAX, 1U)) {
        return false;
    }
    for (uint32_t owner = 0U; owner < kHostOwnerCount; ++owner) {
        float expected_slot0 = kHostWorkloadSentinel;
        float expected_slot1 = kHostWorkloadSentinel;
        if (owner < 32U) {
            if (state.roles[owner].completed_by_kind[static_cast<uint32_t>(HostTaskKind::Qk)] != 0U) {
                expected_slot0 = 768.0F;
            }
            if (state.roles[owner].completed_by_kind[static_cast<uint32_t>(HostTaskKind::Pv)] != 0U) {
                expected_slot1 = 768.0F;
            }
        } else if (!g0::IsBuilderOwner(owner, builder_count)) {
            if (state.roles[owner].completed_by_kind[static_cast<uint32_t>(HostTaskKind::Sf)] != 0U) {
                expected_slot0 = 5.0F;
            }
            if (state.roles[owner].completed_by_kind[static_cast<uint32_t>(HostTaskKind::Up)] != 0U) {
                expected_slot1 = 6.0F;
            }
        }
        const uint32_t first_tile = kHostWorkloadInputTiles + owner * kHostWorkloadOutputTilesPerOwner;
        if (!validate_tile(first_tile, expected_slot0, "workspace-owner-slot0", owner, 0U) ||
            !validate_tile(first_tile + 1U, expected_slot1, "workspace-owner-slot1", owner, 1U)) {
            return false;
        }
    }
    return true;
}

bool ValidateRun(
    const FullPaState &state, const FullPaState &initial, const std::vector<float> &workspace,
    uint64_t device_state_address, uint64_t nonce, uint32_t batches, uint32_t builder_count, ValidationFailure *failure
) {
    if (!ValidateFixedRegions(state, initial, nonce, batches, failure)) {
        return false;
    }
    std::vector<uint64_t> task_bases;
    if (!ValidateHeapAndCollectBases(state, initial, batches, &task_bases, failure)) {
        return false;
    }
    const uint32_t task_count = batches * kHostTasksPerBatch;
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        if (!ValidateTask(state, initial, nonce, batches, builder_count, task_bases, task_id, failure)) {
            return false;
        }
    }
    return ValidateBuilderThreads(state, initial, nonce, task_count, builder_count, failure) &&
           ValidateDispatchAndTokens(state, initial, device_state_address, batches, builder_count, failure) &&
           ValidateRolesAndDrain(state, initial, nonce, batches, builder_count, failure) &&
           ValidateWorkspace(state, workspace, builder_count, failure);
}

#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
bool TraceAtomicSiteAllowed(g0_swimlane::TraceDomain domain, g0_swimlane::AtomicSite site) {
    if (site == g0_swimlane::AtomicSite::FatalLoad || site == g0_swimlane::AtomicSite::FatalSet) {
        return true;
    }
    const uint32_t site_id = static_cast<uint32_t>(site);
    return domain == g0_swimlane::TraceDomain::Simt ?
               (site_id >= static_cast<uint32_t>(g0_swimlane::AtomicSite::SimtBuilderStartedIncrement) &&
                    site_id <= static_cast<uint32_t>(g0_swimlane::AtomicSite::SimtBuilderFinishedPublish)) ||
                   site == g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll ||
                   site == g0_swimlane::AtomicSite::SimtMetadataLastWriterLoad :
               site_id >= static_cast<uint32_t>(g0_swimlane::AtomicSite::ScalarDispatchTicket) &&
                   site_id <= static_cast<uint32_t>(g0_swimlane::AtomicSite::ScalarRootFinishedPublish);
}

bool ValidateTraceRecord(
    const g0_swimlane::TraceRecord &record, g0_swimlane::TraceDomain domain, uint32_t task_count,
    uint32_t writer, uint32_t slot, uint64_t role_begin, uint64_t role_end,
    uint64_t *atomic_calls, uint64_t *poll_calls, uint64_t *dcci_calls, uint64_t *dcci_lines,
    uint32_t *poll_records, uint32_t *dcci_records, ValidationFailure *failure
) {
    if (!RecordCheck(
            record.begin <= record.end, failure, "swimlane-log-time-order", record.task_id, writer, slot,
            record.begin, record.end
        ) ||
        !RecordCheck(
            record.task_id == g0_swimlane::kTraceNoTask || record.task_id < task_count, failure,
            "swimlane-log-task", record.task_id, writer, slot, task_count, record.task_id
        )) {
        return false;
    }
    if (domain == g0_swimlane::TraceDomain::Scalar &&
        !RecordCheck(
            role_begin <= record.begin && record.end <= role_end, failure, "swimlane-scalar-log-envelope",
            record.task_id, writer, slot
        )) {
        return false;
    }

    if (record.kind == g0_swimlane::TraceKind::Atomic) {
        const auto site = static_cast<g0_swimlane::AtomicSite>(record.site);
        const auto op = static_cast<g0_swimlane::AtomicOp>(record.op);
        const bool result_used = (record.flags & g0_swimlane::kAtomicResultUsed) != 0U;
        const bool return_ready = (record.flags & g0_swimlane::kAtomicReturnReady) != 0U;
        const bool poll_batch = (record.flags & g0_swimlane::kAtomicPollBatch) != 0U;
        constexpr uint32_t kAllowedAtomicFlags =
            g0_swimlane::kAtomicResultUsed | g0_swimlane::kAtomicReturnReady |
            g0_swimlane::kAtomicValueZero | g0_swimlane::kAtomicPollBatch;
        if (!RecordCheck(
                record.site < static_cast<uint16_t>(g0_swimlane::AtomicSite::Count), failure,
                "swimlane-atomic-site", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                record.op < static_cast<uint8_t>(g0_swimlane::AtomicOp::Count), failure,
                "swimlane-atomic-op", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                TraceAtomicSiteAllowed(domain, site), failure, "swimlane-atomic-domain", record.task_id,
                writer, slot
            ) ||
            !RecordCheck(
                op == g0_swimlane::AtomicSiteExpectedOp(site), failure, "swimlane-atomic-site-op",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                (record.flags & ~kAllowedAtomicFlags) == 0U, failure, "swimlane-atomic-flags",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                !return_ready || result_used, failure, "swimlane-atomic-return-without-result",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                domain == g0_swimlane::TraceDomain::Scalar ? (!result_used || return_ready) : !return_ready,
                failure, "swimlane-atomic-boundary", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                poll_batch == g0_swimlane::AtomicSiteIsPoll(site), failure, "swimlane-atomic-poll-site",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                poll_batch ? (result_used && record.call_count > 0U) : record.call_count == 1U, failure,
                "swimlane-atomic-call-count", record.task_id, writer, slot
            )) {
            return false;
        }
        *atomic_calls += record.call_count;
        if (poll_batch) {
            *poll_calls += record.call_count;
            ++*poll_records;
        }
        return true;
    }

    if (record.kind == g0_swimlane::TraceKind::Dcci) {
        const auto site = static_cast<g0_swimlane::DcciSite>(record.site);
        const auto op = static_cast<g0_swimlane::DcciOp>(record.op);
        const uint32_t lines = g0_swimlane::DcciLineCount(record.flags);
        if (!RecordCheck(
                domain == g0_swimlane::TraceDomain::Scalar, failure, "swimlane-dcci-domain",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                record.site < static_cast<uint16_t>(g0_swimlane::DcciSite::Count), failure,
                "swimlane-dcci-site", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                record.op < static_cast<uint8_t>(g0_swimlane::DcciOp::Count), failure,
                "swimlane-dcci-op", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                op == g0_swimlane::DcciOp::Invalidate, failure, "swimlane-dcci-site-op",
                record.task_id, writer, slot
            ) ||
            !RecordCheck(
                (record.flags & g0_swimlane::kDcciTrailingDsb) != 0U &&
                    (record.flags & ((1U << g0_swimlane::kDcciLineCountShift) - 1U)) ==
                        g0_swimlane::kDcciTrailingDsb,
                failure, "swimlane-dcci-flags", record.task_id, writer, slot
            ) ||
            !RecordCheck(
                record.call_count > 0U && lines >= record.call_count && lines <= g0_swimlane::kDcciLineCountMax,
                failure, "swimlane-dcci-counts", record.task_id, writer, slot
            )) {
            return false;
        }
        (void)site;
        *dcci_calls += record.call_count;
        *dcci_lines += lines;
        ++*dcci_records;
        return true;
    }
    return RecordCheck(false, failure, "swimlane-log-kind", record.task_id, writer, slot);
}

bool ValidateTraceLog(
    const g0_swimlane::TraceLogControl &control, const g0_swimlane::TraceRecord *records,
    const g0_swimlane::TraceRecord *initial_records, g0_swimlane::TraceDomain domain, uint32_t capacity,
    uint64_t nonce, uint32_t task_count, uint32_t builder_count, uint32_t writer,
    uint64_t role_begin, uint64_t role_end,
    ValidationFailure *failure
) {
    if (!RecordCheck(control.launch_nonce == nonce, failure, "swimlane-log-nonce", UINT32_MAX, writer) ||
        !RecordCheck(control.writer_id == writer, failure, "swimlane-log-writer", UINT32_MAX, writer) ||
        !RecordCheck(control.domain == domain, failure, "swimlane-log-domain", UINT32_MAX, writer) ||
        !RecordCheck(
            control.record_count > 0U && control.record_count <= capacity, failure, "swimlane-log-record-count",
            UINT32_MAX, writer, UINT32_MAX, capacity, control.record_count
        ) ||
        !RecordCheck(
            control.dropped_records == 0U, failure, "swimlane-log-dropped", UINT32_MAX, writer,
            UINT32_MAX, 0U, control.dropped_records
        )) {
        return false;
    }
    uint64_t atomic_calls = 0U;
    uint64_t poll_calls = 0U;
    uint64_t dcci_calls = 0U;
    uint64_t dcci_lines = 0U;
    uint32_t poll_records = 0U;
    uint32_t dcci_records = 0U;
    for (uint32_t slot = 0U; slot < control.record_count; ++slot) {
        if (!ValidateTraceRecord(
                records[slot], domain, task_count, writer, slot, role_begin, role_end, &atomic_calls,
                &poll_calls, &dcci_calls, &dcci_lines, &poll_records, &dcci_records, failure
            )) {
            return false;
        }
        if (domain == g0_swimlane::TraceDomain::Simt && records[slot].task_id != g0_swimlane::kTraceNoTask &&
            !RecordCheck(
                g0::BuilderThreadForTask(records[slot].task_id, builder_count) / g0::kWarpSize == writer, failure,
                "swimlane-simt-log-writer", records[slot].task_id, writer, slot,
                g0::BuilderThreadForTask(records[slot].task_id, builder_count) / g0::kWarpSize, writer
            )) {
            return false;
        }
    }
    for (uint32_t slot = control.record_count; slot < capacity; ++slot) {
        if (!RecordCheck(
                std::memcmp(&records[slot], &initial_records[slot], sizeof(records[slot])) == 0, failure,
                "swimlane-log-tail-mutated", UINT32_MAX, writer, slot
            )) {
            return false;
        }
    }
    return RecordCheck(
               control.dcci_calls == dcci_calls, failure, "swimlane-log-dcci-total", UINT32_MAX, writer,
               UINT32_MAX, dcci_calls, control.dcci_calls
           ) &&
           RecordCheck(
               control.dcci_lines == dcci_lines, failure, "swimlane-log-dcci-lines", UINT32_MAX, writer,
               UINT32_MAX, dcci_lines, control.dcci_lines
           ) &&
           RecordCheck(
               control.dcci_records == dcci_records, failure, "swimlane-log-dcci-records", UINT32_MAX, writer,
               UINT32_MAX, dcci_records, control.dcci_records
           );
}

bool ValidateSwimlaneTrace(
    const LaunchState &state, const LaunchState &initial, uint64_t nonce, uint32_t batches, uint32_t builder_count,
    ValidationFailure *failure
) {
    const g0_swimlane::TraceState &trace = state.trace;
    const g0_swimlane::TraceState &initial_trace = initial.trace;
    const uint32_t task_count = g0::TaskCount(batches);
    if (!RecordCheck(
            std::memcmp(&trace.control, &initial_trace.control, sizeof(trace.control)) == 0, failure,
            "swimlane-control-mutated"
        )) {
        return false;
    }
    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        const g0_swimlane::RoleTrace &role = trace.roles[owner];
        const uint32_t expected_subblock = owner < g0::kAicOwnerCount ? 0U : (owner - g0::kAicOwnerCount) % 2U;
        if (!RecordCheck(role.launch_nonce == nonce, failure, "swimlane-role-nonce", UINT32_MAX, owner) ||
            !RecordCheck(role.owner == owner, failure, "swimlane-role-owner", UINT32_MAX, owner) ||
            !RecordCheck(
                role.role == static_cast<uint32_t>(g0::OwnerRoleAt(owner, builder_count)), failure,
                "swimlane-role-kind", UINT32_MAX, owner
            ) ||
            !RecordCheck(
                role.physical_block == g0::OwnerPhysicalBlock(owner), failure, "swimlane-role-block", UINT32_MAX,
                owner
            ) ||
            !RecordCheck(
                role.subblock == expected_subblock, failure, "swimlane-role-subblock", UINT32_MAX, owner
            ) ||
            !RecordCheck(
                role.entry <= role.config_ready && role.config_ready <= role.work_begin &&
                    role.work_begin <= role.work_end && role.work_end <= role.drain_begin &&
                    role.drain_begin <= role.drain_end && role.drain_end <= role.exit,
                failure, "swimlane-role-time-order", UINT32_MAX, owner
            )) {
            return false;
        }
    }

    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const g0_swimlane::BuilderTaskTrace &builder = trace.builders[task_id];
        const uint32_t expected_thread = g0::BuilderThreadForTask(task_id, builder_count);
        if (!RecordCheck(builder.launch_nonce == nonce, failure, "swimlane-builder-nonce", task_id) ||
            !RecordCheck(builder.task_id == task_id, failure, "swimlane-builder-task", task_id) ||
            !RecordCheck(
                builder.builder_thread == expected_thread, failure, "swimlane-builder-thread", task_id,
                UINT32_MAX, UINT32_MAX, expected_thread, builder.builder_thread
            ) ||
            !RecordCheck(
                g0::IsBuilderOwner(builder.build_owner, builder_count), failure, "swimlane-builder-owner", task_id
            ) ||
            !RecordCheck(
                builder.attempt_begin <= builder.claim_end && builder.claim_end <= builder.prepare_fence1 &&
                    builder.prepare_fence1 <= builder.prepare_fence2 && builder.prepare_fence2 <= builder.prepare_end &&
                    builder.prepare_end <= builder.commit_begin && builder.commit_begin <= builder.commit_poll_end &&
                    builder.commit_poll_end <= builder.commit_insert_end && builder.commit_insert_end <= builder.commit_end &&
                    builder.commit_end <= builder.report_end,
                failure, "swimlane-builder-time-order", task_id
            )) {
            return false;
        }
        const g0::TaskKind kind = g0::TaskKindAt(task_id);
        const g0_swimlane::ExecutorTaskTrace &executor = trace.executors[task_id];
        if (!g0::TaskExecutable(kind)) {
            if (!RecordCheck(
                    std::memcmp(&executor, &initial_trace.executors[task_id], sizeof(executor)) == 0, failure,
                    "swimlane-alloc-executor-mutated", task_id
                )) {
                return false;
            }
            continue;
        }
        if (!RecordCheck(executor.launch_nonce == nonce, failure, "swimlane-executor-nonce", task_id) ||
            !RecordCheck(executor.task_id == task_id, failure, "swimlane-executor-task", task_id) ||
            !RecordCheck(
                executor.task_kind == static_cast<uint32_t>(kind), failure, "swimlane-executor-kind", task_id
            ) ||
            !RecordCheck(
                g0::OwnerCanExecute(executor.execute_owner, g0::TaskEngine(kind), builder_count), failure,
                "swimlane-executor-owner", task_id, executor.execute_owner
            ) ||
            !RecordCheck(
                executor.phase_bits == g0_swimlane::kExpectedExecutorTraceBits, failure,
                "swimlane-executor-phases", task_id, executor.execute_owner
            ) ||
            !RecordCheck(
                executor.ticket_assigned <= executor.claim_end && executor.claim_end <= executor.fanin_ready &&
                    executor.fanin_ready <= executor.execute_begin && executor.execute_begin <= executor.execute_end,
                failure, "swimlane-executor-time-order", task_id, executor.execute_owner
            )) {
            return false;
        }
        const g0_swimlane::RoleTrace &executor_role = trace.roles[executor.execute_owner];
        if (!RecordCheck(
                executor_role.work_begin <= executor.ticket_assigned && executor.execute_end <= executor_role.work_end,
                failure, "swimlane-executor-envelope", task_id, executor.execute_owner
            )) {
            return false;
        }
    }
    for (uint32_t task_id = task_count; task_id < g0_swimlane::kTraceTaskCapacity; ++task_id) {
        if (!RecordCheck(
                std::memcmp(&trace.builders[task_id], &initial_trace.builders[task_id], sizeof(trace.builders[task_id])) ==
                    0,
                failure, "swimlane-inactive-builder-mutated", task_id
            ) ||
            !RecordCheck(
                std::memcmp(
                    &trace.executors[task_id], &initial_trace.executors[task_id], sizeof(trace.executors[task_id])
                ) == 0,
                failure, "swimlane-inactive-executor-mutated", task_id
            )) {
            return false;
        }
    }
    const uint32_t active_simt_writers = g0::BuilderLeaderCount(builder_count);
    for (uint32_t writer = 0U; writer < active_simt_writers; ++writer) {
        if (!ValidateTraceLog(
                trace.simt_logs[writer], &trace.simt_records[writer][0], &initial_trace.simt_records[writer][0],
                g0_swimlane::TraceDomain::Simt, g0_swimlane::kTraceSimtRecordsPerWriter, nonce, task_count,
                builder_count, writer, 0U, UINT64_MAX, failure
            )) {
            return false;
        }
    }
    for (uint32_t writer = active_simt_writers; writer < g0_swimlane::kTraceSimtWriterCount; ++writer) {
        if (!RecordCheck(
                std::memcmp(&trace.simt_logs[writer], &initial_trace.simt_logs[writer],
                            sizeof(trace.simt_logs[writer])) == 0,
                failure, "swimlane-inactive-simt-log-mutated", UINT32_MAX, writer
            ) ||
            !RecordCheck(
                std::memcmp(&trace.simt_records[writer][0], &initial_trace.simt_records[writer][0],
                            sizeof(trace.simt_records[writer])) == 0,
                failure, "swimlane-inactive-simt-records-mutated", UINT32_MAX, writer
            )) {
            return false;
        }
    }
    for (uint32_t writer = 0U; writer < g0_swimlane::kTraceScalarWriterCount; ++writer) {
        if (!ValidateTraceLog(
                trace.scalar_logs[writer], &trace.scalar_records[writer][0],
                &initial_trace.scalar_records[writer][0], g0_swimlane::TraceDomain::Scalar,
                g0_swimlane::kTraceScalarRecordsPerWriter, nonce, task_count, builder_count, writer,
                trace.roles[writer].entry, trace.roles[writer].exit, failure
            )) {
            return false;
        }
    }
    return true;
}

const char *TraceTaskKindName(g0::TaskKind kind) {
    switch (kind) {
    case g0::TaskKind::Alloc:
        return "Alloc";
    case g0::TaskKind::Qk:
        return "QK";
    case g0::TaskKind::Sf:
        return "SF";
    case g0::TaskKind::Pv:
        return "PV";
    case g0::TaskKind::Up:
        return "UP";
    case g0::TaskKind::Count:
        return "Count";
    }
    return "Unknown";
}

const char *TraceRoleName(g0::OwnerRole role) {
    switch (role) {
    case g0::OwnerRole::AicExecutor:
        return "AIC executor";
    case g0::OwnerRole::AivBuilder:
        return "AIV SIMT builder";
    case g0::OwnerRole::AivExecutor:
        return "AIV executor";
    }
    return "unknown role";
}

const char *TraceDomainName(g0_swimlane::TraceDomain domain) {
    return domain == g0_swimlane::TraceDomain::Scalar ? "scalar" : "simt";
}

const char *TraceAtomicOpName(g0_swimlane::AtomicOp op) {
    switch (op) {
    case g0_swimlane::AtomicOp::Load:
        return "load";
    case g0_swimlane::AtomicOp::Exchange:
        return "exchange";
    case g0_swimlane::AtomicOp::FetchAdd:
        return "fetch_add";
    case g0_swimlane::AtomicOp::CompareExchange:
        return "compare_exchange";
    case g0_swimlane::AtomicOp::Count:
        return "count";
    }
    return "unknown";
}

const char *TraceAtomicSiteName(g0_swimlane::AtomicSite site) {
    switch (site) {
    case g0_swimlane::AtomicSite::FatalLoad:
        return "fatal_load";
    case g0_swimlane::AtomicSite::FatalSet:
        return "fatal_set";
    case g0_swimlane::AtomicSite::SimtBuilderStartedIncrement:
        return "simt_builder_started_increment";
    case g0_swimlane::AtomicSite::SimtBuilderStartedPoll:
        return "simt_builder_started_poll";
    case g0_swimlane::AtomicSite::SimtTaskBuildAttemptIncrement:
        return "simt_task_build_attempt_increment";
    case g0_swimlane::AtomicSite::SimtTaskBuildPreparedIncrement:
        return "simt_task_build_prepared_increment";
    case g0_swimlane::AtomicSite::SimtTaskBuildClaim:
        return "simt_task_build_claim";
    case g0_swimlane::AtomicSite::SimtHeapShardReserve:
        return "simt_heap_shard_reserve";
    case g0_swimlane::AtomicSite::SimtHeapVendReserve:
        return "simt_heap_vend_reserve";
    case g0_swimlane::AtomicSite::SimtHeapVendLoad:
        return "simt_heap_vend_load";
    case g0_swimlane::AtomicSite::SimtTaskBasePublish:
        return "simt_task_base_publish";
    case g0_swimlane::AtomicSite::SimtCompletionVendReportPublish:
        return "simt_completion_vend_report_publish";
    case g0_swimlane::AtomicSite::SimtOutputPublishedPublish:
        return "simt_output_published_publish";
    case g0_swimlane::AtomicSite::SimtOutputLastWriterPublish:
        return "simt_output_last_writer_publish";
    case g0_swimlane::AtomicSite::SimtProducerTaskBasePoll:
        return "simt_producer_task_base_poll";
    case g0_swimlane::AtomicSite::SimtInsertPredecessorPoll:
        return "simt_insert_predecessor_poll";
    case g0_swimlane::AtomicSite::SimtMetadataLastWriterCommit:
        return "simt_metadata_last_writer_commit";
    case g0_swimlane::AtomicSite::SimtInsertCompletionPublish:
        return "simt_insert_completion_publish";
    case g0_swimlane::AtomicSite::SimtAllocCompletionVendPublish:
        return "simt_alloc_completion_vend_publish";
    case g0_swimlane::AtomicSite::SimtAllocCompletionFlagPublish:
        return "simt_alloc_completion_flag_publish";
    case g0_swimlane::AtomicSite::SimtAllocDoneIncrement:
        return "simt_alloc_done_increment";
    case g0_swimlane::AtomicSite::SimtExecBuiltPublish:
        return "simt_exec_built_publish";
    case g0_swimlane::AtomicSite::SimtBuildReportPublish:
        return "simt_build_report_publish";
    case g0_swimlane::AtomicSite::SimtBuilderFinishedPublish:
        return "simt_builder_finished_publish";
    case g0_swimlane::AtomicSite::ScalarDispatchTicket:
        return "scalar_dispatch_ticket";
    case g0_swimlane::AtomicSite::ScalarProducerTaskBaseLoad:
        return "scalar_producer_task_base_load";
    case g0_swimlane::AtomicSite::ScalarExecStatePoll:
        return "scalar_exec_state_poll";
    case g0_swimlane::AtomicSite::ScalarExecClaim:
        return "scalar_exec_claim";
    case g0_swimlane::AtomicSite::ScalarFaninFlagPoll:
        return "scalar_fanin_flag_poll";
    case g0_swimlane::AtomicSite::ScalarExecutionWitnessPublish:
        return "scalar_execution_witness_publish";
    case g0_swimlane::AtomicSite::ScalarCompletionVendPublish:
        return "scalar_completion_vend_publish";
    case g0_swimlane::AtomicSite::ScalarCompletionFlagPublish:
        return "scalar_completion_flag_publish";
    case g0_swimlane::AtomicSite::ScalarExecDonePublish:
        return "scalar_exec_done_publish";
    case g0_swimlane::AtomicSite::ScalarDoneCountIncrement:
        return "scalar_done_count_increment";
    case g0_swimlane::AtomicSite::ScalarEngineDoneIncrement:
        return "scalar_engine_done_increment";
    case g0_swimlane::AtomicSite::ScalarDrainArrive:
        return "scalar_drain_arrive";
    case g0_swimlane::AtomicSite::ScalarDrainArrivalPoll:
        return "scalar_drain_arrival_poll";
    case g0_swimlane::AtomicSite::ScalarDrainVerifyLoad:
        return "scalar_drain_verify_load";
    case g0_swimlane::AtomicSite::ScalarRootFinishedPublish:
        return "scalar_root_finished_publish";
    case g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll:
        return "simt_metadata_output_published_poll";
    case g0_swimlane::AtomicSite::SimtMetadataLastWriterLoad:
        return "simt_metadata_last_writer_load";
    case g0_swimlane::AtomicSite::Count:
        return "count";
    }
    return "unknown";
}

const char *TraceDcciSiteName(g0_swimlane::DcciSite site) {
    switch (site) {
    case g0_swimlane::DcciSite::StartupConfigInvalidate:
        return "startup_config";
    case g0_swimlane::DcciSite::DispatchTaskIdInvalidate:
        return "dispatch_task_id";
    case g0_swimlane::DcciSite::ExecPayloadInvalidate:
        return "exec_payload";
    case g0_swimlane::DcciSite::TerminalTokenInvalidate:
        return "terminal_token";
    case g0_swimlane::DcciSite::Count:
        return "count";
    }
    return "unknown";
}

const char *TraceDcciOpName(g0_swimlane::DcciOp op) {
    switch (op) {
    case g0_swimlane::DcciOp::Invalidate:
        return "invalidate";
    case g0_swimlane::DcciOp::CleanOut:
        return "clean_out";
    case g0_swimlane::DcciOp::Count:
        return "count";
    }
    return "unknown";
}

void WriteTraceSeparator(std::ofstream *output, bool *first) {
    if (!*first) {
        *output << ",\n";
    }
    *first = false;
}

void WriteTraceMetadata(
    std::ofstream *output, bool *first, const char *name, uint32_t pid, uint32_t tid, const std::string &value
) {
    WriteTraceSeparator(output, first);
    *output << "{\"name\":\"" << name << "\",\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":" << tid
            << ",\"args\":{\"name\":\"" << value << "\"}}";
}

void WriteTraceEvent(
    std::ofstream *output, bool *first, const std::string &name, const char *category, uint32_t pid, uint32_t tid,
    uint64_t begin, uint64_t end, uint64_t origin, uint32_t task_id = UINT32_MAX, uint32_t owner = UINT32_MAX,
    uint32_t polls = UINT32_MAX
) {
    WriteTraceSeparator(output, first);
    const long double begin_us = static_cast<long double>(begin - origin) / 1000.0L;
    const long double duration_us = static_cast<long double>(end - begin) / 1000.0L;
    *output << std::fixed << std::setprecision(3) << "{\"name\":\"" << name << "\",\"cat\":\"" << category
            << "\",\"ph\":\"X\",\"pid\":" << pid << ",\"tid\":" << tid << ",\"ts\":" << begin_us
            << ",\"dur\":" << duration_us << ",\"args\":{";
    bool first_argument = true;
    const auto argument = [&](const char *key, uint32_t value) {
        if (!first_argument) {
            *output << ',';
        }
        *output << "\"" << key << "\":" << value;
        first_argument = false;
    };
    if (task_id != UINT32_MAX) {
        argument("task", task_id);
        argument("batch", task_id / g0::kTasksPerBatch);
    }
    if (owner != UINT32_MAX) {
        argument("owner", owner);
    }
    if (polls != UINT32_MAX) {
        argument("polls", polls);
    }
    *output << "}}";
}

void WriteProfileTraceEvent(
    std::ofstream *output, bool *first, const g0_swimlane::TraceRecord &record,
    g0_swimlane::TraceDomain domain, uint32_t writer, uint32_t pid, uint32_t tid,
    uint64_t begin, uint64_t end, uint64_t origin
) {
    const bool has_task = record.task_id != g0_swimlane::kTraceNoTask;
    const uint64_t raw_ticks = record.end - record.begin;
    std::string name;
    std::string category;
    if (record.kind == g0_swimlane::TraceKind::Atomic) {
        const auto site = static_cast<g0_swimlane::AtomicSite>(record.site);
        const auto op = static_cast<g0_swimlane::AtomicOp>(record.op);
        const bool poll_batch = (record.flags & g0_swimlane::kAtomicPollBatch) != 0U;
        const bool return_ready = (record.flags & g0_swimlane::kAtomicReturnReady) != 0U;
        const std::string boundary = return_ready ? "return_ready" : "source_issue";
        name = poll_batch ? "atomic.poll_batch." + boundary + "." : "atomic." + boundary + ".";
        name += TraceAtomicSiteName(site);
        name += ".";
        name += TraceAtomicOpName(op);
        if (poll_batch) {
            name += "x" + std::to_string(record.call_count);
        } else if (has_task) {
            name += "#" + std::to_string(record.task_id);
        }
        category = poll_batch ? "atomic.poll_batch" : "atomic." + boundary;
    } else {
        const auto site = static_cast<g0_swimlane::DcciSite>(record.site);
        const auto op = static_cast<g0_swimlane::DcciOp>(record.op);
        name = "dcci." + std::string(TraceDcciSiteName(site)) + "." + TraceDcciOpName(op) + "x" +
               std::to_string(record.call_count) + ".lines" +
               std::to_string(g0_swimlane::DcciLineCount(record.flags));
        if (has_task) {
            name += "#" + std::to_string(record.task_id);
        }
        category = "dcci";
    }

    WriteTraceSeparator(output, first);
    const long double begin_us = static_cast<long double>(begin - origin) / 1000.0L;
    const long double duration_us = static_cast<long double>(end - begin) / 1000.0L;
    *output << std::fixed << std::setprecision(3) << "{\"name\":\"" << name << "\",\"cat\":\""
            << category << "\",\"ph\":\"X\",\"pid\":" << pid << ",\"tid\":" << tid
            << ",\"ts\":" << begin_us << ",\"dur\":" << duration_us << ",\"args\":{";
    *output << "\"phase\":\"" << (record.kind == g0_swimlane::TraceKind::Atomic ? "atomic" : "dcci")
            << "\",\"execution_unit\":\"" << TraceDomainName(domain) << "\",\"writer\":" << writer;
    if (has_task) {
        *output << ",\"task_id\":" << record.task_id << ",\"batch\":"
                << record.task_id / g0::kTasksPerBatch;
    }
    if (record.kind == g0_swimlane::TraceKind::Atomic) {
        const bool poll_batch = (record.flags & g0_swimlane::kAtomicPollBatch) != 0U;
        *output << ",\"site\":\""
                << TraceAtomicSiteName(static_cast<g0_swimlane::AtomicSite>(record.site))
                << "\",\"site_id\":" << record.site << ",\"op\":\""
                << TraceAtomicOpName(static_cast<g0_swimlane::AtomicOp>(record.op))
                << "\",\"op_id\":" << static_cast<uint32_t>(record.op)
                << ",\"call_count\":" << record.call_count
                << ",\"result_used\":"
                << ((record.flags & g0_swimlane::kAtomicResultUsed) != 0U ? "true" : "false")
                << ",\"return_ready_observed\":"
                << ((record.flags & g0_swimlane::kAtomicReturnReady) != 0U ? "true" : "false")
                << ",\"completion_boundary\":\""
                << ((record.flags & g0_swimlane::kAtomicReturnReady) != 0U ? "return_value_ready" :
                                                                                 "source_issue_bracket")
                << "\",\"is_poll_batch\":" << (poll_batch ? "true" : "false");
        if (poll_batch) {
            *output << ",\"duration_semantics\":\"logical_poll_episode_envelope_not_single_atomic_latency\""
                       ",\"estimate_formula\":\"call_count * calibrated_atomic_cost\"";
        }
        if (static_cast<g0_swimlane::AtomicOp>(record.op) == g0_swimlane::AtomicOp::Load) {
            *output << ",\"value_zero\":"
                    << ((record.flags & g0_swimlane::kAtomicValueZero) != 0U ? "true" : "false");
        }
    } else {
        *output << ",\"site\":\"" << TraceDcciSiteName(static_cast<g0_swimlane::DcciSite>(record.site))
                << "\",\"site_id\":" << record.site << ",\"op\":\""
                << TraceDcciOpName(static_cast<g0_swimlane::DcciOp>(record.op))
                << "\",\"op_id\":" << static_cast<uint32_t>(record.op)
                << ",\"call_count\":" << record.call_count << ",\"cache_line_count\":"
                << g0_swimlane::DcciLineCount(record.flags) << ",\"trailing_dsb\":true";
    }
    *output << ",\"raw_ticks\":" << raw_ticks << ",\"clock_domain\":\""
            << (domain == g0_swimlane::TraceDomain::Scalar ? "scalar_sys_cnt_1ghz" : "simt_clock64")
            << "\",\"flags\":" << record.flags << "}}";
}

bool WriteSwimlaneJson(const LaunchState &state, const std::string &path) {
    std::error_code path_error;
    if (std::filesystem::exists(path, path_error) || path_error) {
        std::fprintf(
            stderr, "refusing to overwrite existing swimlane output: %s%s\n", path.c_str(),
            path_error ? " (path check failed)" : ""
        );
        return false;
    }
    const g0_swimlane::TraceState &trace = state.trace;
    struct TraceTotals {
        uint64_t atomic_calls = 0U;
        uint64_t poll_calls = 0U;
        uint64_t dcci_calls = 0U;
        uint64_t dcci_lines = 0U;
        uint64_t records = 0U;
        uint64_t poll_records = 0U;
        uint64_t dcci_records = 0U;
    } simt_totals, scalar_totals;
    const auto add_control = [](TraceTotals *totals, const g0_swimlane::TraceLogControl &control) {
        totals->atomic_calls += control.atomic_calls;
        totals->poll_calls += control.poll_calls;
        totals->dcci_calls += control.dcci_calls;
        totals->dcci_lines += control.dcci_lines;
        totals->records += control.record_count;
        totals->poll_records += control.poll_records;
        totals->dcci_records += control.dcci_records;
    };
    const uint32_t active_simt_writers = trace.control.simt_writer_count;
    for (uint32_t writer = 0U; writer < active_simt_writers; ++writer) {
        add_control(&simt_totals, trace.simt_logs[writer]);
    }
    for (uint32_t writer = 0U; writer < g0_swimlane::kTraceScalarWriterCount; ++writer) {
        add_control(&scalar_totals, trace.scalar_logs[writer]);
    }

    uint64_t origin = UINT64_MAX;
    uint64_t finish = 0U;
    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        origin = std::min(origin, trace.roles[owner].entry);
        finish = std::max(finish, trace.roles[owner].exit);
    }
    if (origin == UINT64_MAX || finish < origin) {
        return false;
    }
    const uint32_t builder_count = state.full_pa.control.builder_count;
    std::array<uint64_t, g0::kMaxBuilderCount> simt_raw_begin{};
    std::array<uint64_t, g0::kMaxBuilderCount> simt_raw_end{};
    simt_raw_begin.fill(UINT64_MAX);
    const uint32_t task_count = state.full_pa.control.task_count;
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const g0_swimlane::BuilderTaskTrace &builder = trace.builders[task_id];
        const uint32_t builder_instance = builder.build_owner - g0::kBuilderOwner;
        simt_raw_begin[builder_instance] = std::min(simt_raw_begin[builder_instance], builder.attempt_begin);
        simt_raw_end[builder_instance] = std::max(simt_raw_end[builder_instance], builder.report_end);
    }
    for (uint32_t writer = 0U; writer < active_simt_writers; ++writer) {
        const uint32_t builder_instance = writer / g0::kBuilderWarpCount;
        const uint32_t record_count = trace.simt_logs[writer].record_count;
        for (uint32_t slot = 0U; slot < record_count; ++slot) {
            simt_raw_begin[builder_instance] =
                std::min(simt_raw_begin[builder_instance], trace.simt_records[writer][slot].begin);
            simt_raw_end[builder_instance] =
                std::max(simt_raw_end[builder_instance], trace.simt_records[writer][slot].end);
        }
    }
    uint64_t builder_work_begin = UINT64_MAX;
    uint64_t builder_work_end = 0U;
    uint64_t max_simt_raw_span = 0U;
    for (uint32_t builder = 0U; builder < builder_count; ++builder) {
        const g0_swimlane::RoleTrace &builder_role = trace.roles[g0::BuilderOwnerForInstance(builder)];
        builder_work_begin = std::min(builder_work_begin, builder_role.work_begin);
        builder_work_end = std::max(builder_work_end, builder_role.work_end);
        if (simt_raw_begin[builder] == UINT64_MAX || simt_raw_end[builder] <= simt_raw_begin[builder] ||
            builder_role.work_end <= builder_role.work_begin) {
            return false;
        }
        max_simt_raw_span = std::max(max_simt_raw_span, simt_raw_end[builder] - simt_raw_begin[builder]);
    }
    if (builder_work_begin == UINT64_MAX || builder_work_end <= builder_work_begin) {
        return false;
    }
    const auto align_simt_clock = [&](uint64_t raw, uint32_t builder_instance) {
        const g0_swimlane::RoleTrace &builder_role =
            trace.roles[g0::BuilderOwnerForInstance(builder_instance)];
        const long double raw_offset = static_cast<long double>(raw - simt_raw_begin[builder_instance]);
        const long double scalar_span = static_cast<long double>(builder_role.work_end - builder_role.work_begin);
        const long double raw_span =
            static_cast<long double>(simt_raw_end[builder_instance] - simt_raw_begin[builder_instance]);
        return builder_role.work_begin + static_cast<uint64_t>(raw_offset * scalar_span / raw_span + 0.5L);
    };
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "cannot open swimlane output: %s\n", path.c_str());
        return false;
    }
    output << "{\"schema\":\"simt_cross_core_g0_swimlane_v5\","
              "\"clock\":\"Scalar get_sys_cnt: 1 ns/tick; SIMT CLOCK64: raw ticks\","
              "\"simt_alignment\":\"per_builder_affine_to_own_scalar_vf_envelope_for_display_only\","
              "\"simt_atomic_boundary\":\"source_issue; CCEC SIMT return-register dependency is not claimed\","
              "\"instrumented\":true,\"batches\":"
           << state.full_pa.control.batch_count << ",\"tasks\":" << state.full_pa.control.task_count
           << ",\"device_span_ns\":" << (finish - origin) << ",\"simt_clock64_span_ticks\":"
           << max_simt_raw_span << ",\"simt_clock64_span_ticks_by_builder\":[";
    for (uint32_t builder = 0U; builder < builder_count; ++builder) {
        output << (builder == 0U ? "" : ",") << (simt_raw_end[builder] - simt_raw_begin[builder]);
    }
    output << "],\"atomic_dcci_summary\":{"
           << "\"simt\":{\"atomic_calls\":" << simt_totals.atomic_calls << ",\"poll_calls\":"
           << simt_totals.poll_calls << ",\"records\":" << simt_totals.records
           << ",\"poll_records\":" << simt_totals.poll_records << "},\"scalar\":{\"atomic_calls\":"
           << scalar_totals.atomic_calls << ",\"poll_calls\":" << scalar_totals.poll_calls
           << ",\"dcci_calls\":" << scalar_totals.dcci_calls << ",\"dcci_cache_lines\":"
           << scalar_totals.dcci_lines << ",\"records\":" << scalar_totals.records
           << ",\"poll_records\":" << scalar_totals.poll_records << ",\"dcci_records\":"
           << scalar_totals.dcci_records << "}},\"traceEvents\":[\n";
    bool first = true;
    WriteTraceMetadata(&output, &first, "process_name", 1U, 0U, "Main Scalar roles + task/atomic/DCCI overlays");
    WriteTraceMetadata(&output, &first, "process_name", 2U, 0U, "SIMT builder warps + atomic overlays");
    for (uint32_t owner = 0U; owner < g0::kOwnerCount; ++owner) {
        const g0::OwnerRole role = g0::OwnerRoleAt(owner, state.full_pa.control.builder_count);
        const std::string owner_name = std::string(TraceRoleName(role)) + " #" + std::to_string(owner);
        WriteTraceMetadata(&output, &first, "thread_name", 1U, owner, owner_name);
        const g0_swimlane::RoleTrace &record = trace.roles[owner];
        WriteTraceEvent(
            &output, &first, "Scalar role lifetime", "scalar", 1U, owner, record.entry, record.exit, origin,
            UINT32_MAX, owner
        );
        WriteTraceEvent(
            &output, &first, "setup/config", "scalar", 1U, owner, record.entry, record.config_ready, origin,
            UINT32_MAX, owner
        );
        WriteTraceEvent(
            &output, &first, role == g0::OwnerRole::AivBuilder ? "SIMT VF build" : "executor loop", "scalar", 1U,
            owner, record.work_begin, record.work_end, origin, UINT32_MAX, owner
        );
        WriteTraceEvent(
            &output, &first, "final_drain", "scalar", 1U, owner, record.drain_begin, record.drain_end, origin,
            UINT32_MAX, owner
        );
    }
    for (uint32_t writer = 0U; writer < active_simt_writers; ++writer) {
        const uint32_t builder = writer / g0::kBuilderWarpCount;
        const uint32_t warp = writer % g0::kBuilderWarpCount;
        WriteTraceMetadata(
            &output, &first, "thread_name", 2U, writer,
            "AIV" + std::to_string(builder) + " SIMT warp " + std::to_string(warp)
        );
    }
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const g0_swimlane::BuilderTaskTrace &builder = trace.builders[task_id];
        const g0::TaskKind kind = g0::TaskKindAt(task_id);
        const uint32_t builder_instance = builder.build_owner - g0::kBuilderOwner;
        const uint32_t warp = builder.builder_thread / g0::kWarpSize;
        const std::string prefix = "task[" + std::to_string(task_id) + "] " + TraceTaskKindName(kind);
        const uint64_t attempt_begin = align_simt_clock(builder.attempt_begin, builder_instance);
        const uint64_t claim_end = align_simt_clock(builder.claim_end, builder_instance);
        const uint64_t prepare_end = align_simt_clock(builder.prepare_end, builder_instance);
        const uint64_t prepare_fence1 = align_simt_clock(builder.prepare_fence1, builder_instance);
        const uint64_t prepare_fence2 = align_simt_clock(builder.prepare_fence2, builder_instance);
        const uint64_t commit_begin = align_simt_clock(builder.commit_begin, builder_instance);
        const uint64_t commit_end = align_simt_clock(builder.commit_end, builder_instance);
        const uint64_t commit_poll_end = align_simt_clock(builder.commit_poll_end, builder_instance);
        const uint64_t commit_insert_end = align_simt_clock(builder.commit_insert_end, builder_instance);
        const uint64_t report_end = align_simt_clock(builder.report_end, builder_instance);
        WriteTraceEvent(
            &output, &first, prefix + " build", "simt_build", 2U, warp, attempt_begin, report_end, origin, task_id,
            builder.build_owner, builder.insert_poll_count
        );
        WriteTraceEvent(
            &output, &first, "claim", "simt_build", 2U, warp, attempt_begin, claim_end, origin, task_id,
            builder.build_owner
        );
        WriteTraceEvent(
            &output, &first, "prepare_phase1", "simt_build", 2U, warp, claim_end, prepare_fence1, origin, task_id,
            builder.build_owner
        );
        WriteTraceEvent(
            &output, &first, "prepare_phase2", "simt_build", 2U, warp, prepare_fence1, prepare_fence2, origin, task_id,
            builder.build_owner
        );
        WriteTraceEvent(
            &output, &first, "prepare_phase3", "simt_build", 2U, warp, prepare_fence2, prepare_end, origin, task_id,
            builder.build_owner
        );
        WriteTraceEvent(
            &output, &first, "commit_poll", "simt_build", 2U, warp, commit_begin, commit_poll_end, origin, task_id,
            builder.build_owner, builder.insert_poll_count
        );
        WriteTraceEvent(
            &output, &first, "commit_work", "simt_build", 2U, warp, commit_poll_end, commit_insert_end, origin, task_id,
            builder.build_owner, builder.insert_poll_count
        );
        WriteTraceEvent(
            &output, &first, "commit_post", "simt_build", 2U, warp, commit_insert_end, commit_end, origin, task_id,
            builder.build_owner
        );
        WriteTraceEvent(
            &output, &first, "build_report_publish", "simt_build", 2U, warp, commit_end, report_end, origin,
            task_id, builder.build_owner
        );
        if (!g0::TaskExecutable(kind)) {
            continue;
        }
        const g0_swimlane::ExecutorTaskTrace &executor = trace.executors[task_id];
        WriteTraceEvent(
            &output, &first, prefix + " lifecycle", "task", 1U, executor.execute_owner, executor.ticket_assigned,
            executor.execute_end, origin, task_id, executor.execute_owner
        );
        WriteTraceEvent(
            &output, &first, "wait_built+claim", "task_wait", 1U, executor.execute_owner,
            executor.ticket_assigned, executor.claim_end, origin, task_id, executor.execute_owner
        );
        WriteTraceEvent(
            &output, &first, "bind+fanin_wait", "task_wait", 1U, executor.execute_owner, executor.claim_end,
            executor.fanin_ready, origin, task_id, executor.execute_owner
        );
        WriteTraceEvent(
            &output, &first, "task_execute", "task_execute", 1U, executor.execute_owner, executor.execute_begin,
            executor.execute_end, origin, task_id, executor.execute_owner
        );
    }
    for (uint32_t writer = 0U; writer < active_simt_writers; ++writer) {
        const uint32_t builder_instance = writer / g0::kBuilderWarpCount;
        for (uint32_t slot = 0U; slot < trace.simt_logs[writer].record_count; ++slot) {
            const g0_swimlane::TraceRecord &record = trace.simt_records[writer][slot];
            WriteProfileTraceEvent(
                &output, &first, record, g0_swimlane::TraceDomain::Simt, writer, 2U, writer,
                align_simt_clock(record.begin, builder_instance),
                align_simt_clock(record.end, builder_instance), origin
            );
        }
    }
    for (uint32_t writer = 0U; writer < g0_swimlane::kTraceScalarWriterCount; ++writer) {
        for (uint32_t slot = 0U; slot < trace.scalar_logs[writer].record_count; ++slot) {
            const g0_swimlane::TraceRecord &record = trace.scalar_records[writer][slot];
            WriteProfileTraceEvent(
                &output, &first, record, g0_swimlane::TraceDomain::Scalar, writer, 1U, writer,
                record.begin, record.end, origin
            );
        }
    }
    output << "\n]}\n";
    if (!output.good()) {
        std::fprintf(stderr, "failed to write swimlane output: %s\n", path.c_str());
        return false;
    }
    std::printf(
        "[SWIMLANE] file=%s instrumented=yes scalar_clock=1ns simt_clock=affine-display "
        "device_span_us=%.3Lf simt_build_span_us=%.3Lf atomic_calls(simt/scalar)=%llu/%llu "
        "dcci_calls=%llu dcci_lines=%llu\n",
        path.c_str(), static_cast<long double>(finish - origin) / 1000.0L,
        static_cast<long double>(builder_work_end - builder_work_begin) / 1000.0L,
        static_cast<unsigned long long>(simt_totals.atomic_calls),
        static_cast<unsigned long long>(scalar_totals.atomic_calls),
        static_cast<unsigned long long>(scalar_totals.dcci_calls),
        static_cast<unsigned long long>(scalar_totals.dcci_lines)
    );
    return true;
}
#endif

bool ValidateLaunchRun(
    const LaunchState &state, const LaunchState &initial, const std::vector<float> &workspace,
    uint64_t device_state_address, uint64_t nonce, uint32_t batches, uint32_t builder_count, ValidationFailure *failure
) {
    if (!ValidateRun(
            FullPaView(state), FullPaView(initial), workspace, device_state_address, nonce, batches, builder_count,
            failure
        )) {
        return false;
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    return ValidateSwimlaneTrace(state, initial, nonce, batches, builder_count, failure);
#elif defined(SIMT_CROSS_CORE_U2)
    return ValidateU2Staging(state, initial, nonce, batches, failure);
#else
    return true;
#endif
}

bool ParseUnsigned(const char *raw, uint64_t maximum, uint64_t *value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed > maximum) {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--kernel") == 0 && index + 1 < argc) {
            options->kernel_path = argv[++index];
        } else if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), &value)) {
                std::fprintf(stderr, "invalid --device value: %s\n", argv[index]);
                return false;
            }
            options->device = static_cast<int32_t>(value);
        } else if (std::strcmp(argv[index], "--batches") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], 256U, &value) || (value != 1U && value != 256U)) {
                std::fprintf(stderr, "invalid --batches value: %s (expected 1 or 256)\n", argv[index]);
                return false;
            }
            options->batches = static_cast<uint32_t>(value);
        } else if (std::strcmp(argv[index], "--runs") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], 100000U, &value) || value == 0U) {
                std::fprintf(stderr, "invalid --runs value: %s\n", argv[index]);
                return false;
            }
            options->runs = static_cast<uint32_t>(value);
        } else if (std::strcmp(argv[index], "--builders") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            if (!ParseUnsigned(argv[++index], g0::kMaxBuilderCount, &value) ||
                !g0::BuilderCountValid(static_cast<uint32_t>(value))) {
                std::fprintf(
                    stderr, "invalid --builders value: %s (expected 1..%u)\n", argv[index], g0::kMaxBuilderCount
                );
                return false;
            }
#elif defined(SIMT_CROSS_CORE_U2)
            if (!ParseUnsigned(argv[++index], u2::kBuilderCount, &value) || value != u2::kBuilderCount) {
                std::fprintf(stderr, "invalid --builders value: %s (U2 requires 1)\n", argv[index]);
                return false;
            }
#else
            if (!ParseUnsigned(argv[++index], g0::kMaxBuilderCount, &value) ||
                !g0::BuilderCountValid(static_cast<uint32_t>(value))) {
                std::fprintf(
                    stderr, "invalid --builders value: %s (expected 1..%u)\n", argv[index], g0::kMaxBuilderCount
                );
                return false;
            }
#endif
            options->builder_count = static_cast<uint32_t>(value);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        } else if (std::strcmp(argv[index], "--swimlane-json") == 0 && index + 1 < argc) {
            options->swimlane_json = argv[++index];
            if (options->swimlane_json.empty()) {
                std::fprintf(stderr, "--swimlane-json requires a non-empty path\n");
                return false;
            }
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
        } else if (std::strcmp(argv[index], "--acl-config") == 0 && index + 1 < argc) {
            options->acl_config = argv[++index];
            if (options->acl_config.empty()) {
                std::fprintf(stderr, "--acl-config requires a non-empty path\n");
                return false;
            }
#endif
        } else {
#if defined(SIMT_CROSS_CORE_U2)
            const std::string builder_usage = "1";
#else
            const std::string builder_usage = "1.." + std::to_string(g0::kMaxBuilderCount);
#endif
            std::fprintf(
                stderr,
                "usage: %s --kernel FILE [--device N] [--batches 1|256] [--runs N] [--builders %s]"
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
                " [--swimlane-json FILE]"
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
                " --acl-config FILE"
#endif
                "\n",
                argv[0], builder_usage.c_str()
            );
            return false;
        }
    }
    if (options->kernel_path.empty()) {
        std::fprintf(stderr, "--kernel FILE is required\n");
        return false;
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
    if (options->acl_config.empty()) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        std::fprintf(stderr, "G0 swimlane requires --acl-config FILE for its measured SIMT/DVG stack sizes\n");
#else
        std::fprintf(stderr, "U2 requires --acl-config FILE for its measured SIMT/DVG stack sizes\n");
#endif
        return false;
    }
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    if (!options->swimlane_json.empty() && options->runs != 1U) {
        std::fprintf(stderr, "--swimlane-json requires --runs 1 so one file maps to one launch\n");
        return false;
    }
    if (!options->swimlane_json.empty()) {
        std::error_code path_error;
        if (std::filesystem::exists(options->swimlane_json, path_error) || path_error) {
            std::fprintf(
                stderr, "--swimlane-json refuses to overwrite an existing path: %s%s\n",
                options->swimlane_json.c_str(), path_error ? " (path check failed)" : ""
            );
            return false;
        }
    }
#endif
    return true;
}

bool CheckAcl(aclError error, const char *operation) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), operation);
    return false;
}

bool ReadBinary(const std::string &path, std::vector<char> *data) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open kernel binary: %s\n", path.c_str());
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "kernel binary is empty: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data->resize(static_cast<size_t>(size));
    return input.read(data->data(), size).good();
}

class AclSession {
public:
    ~AclSession() {
        if (device_workspace_ != nullptr) {
            (void)aclrtFree(device_workspace_);
        }
        if (device_state_ != nullptr) {
            (void)aclrtFree(device_state_);
        }
        if (binary_ != nullptr) {
            (void)aclrtBinaryUnLoad(binary_);
        }
        if (kernel_end_event_ != nullptr) {
            (void)aclrtDestroyEvent(kernel_end_event_);
        }
        if (kernel_start_event_ != nullptr) {
            (void)aclrtDestroyEvent(kernel_start_event_);
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_set_) {
            (void)aclrtResetDevice(device_);
        }
        if (acl_initialized_) {
            (void)aclFinalize();
        }
    }

    bool Initialize(int32_t device, const std::vector<char> &binary_data, const char *acl_config) {
        device_ = device;
        if (!CheckAcl(aclInit(acl_config), "aclInit")) {
            return false;
        }
        acl_initialized_ = true;
        if (!CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice")) {
            return false;
        }
        device_set_ = true;
        const char *soc_name = aclrtGetSocName();
        if (soc_name == nullptr || soc_name[0] == '\0') {
            std::fprintf(stderr, "aclrtGetSocName returned an empty SoC name\n");
            return false;
        }
        soc_name_ = soc_name;
        if (!CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream") ||
            !CheckAcl(aclrtCreateEvent(&kernel_start_event_), "aclrtCreateEvent(kernel start)") ||
            !CheckAcl(aclrtCreateEvent(&kernel_end_event_), "aclrtCreateEvent(kernel end)")) {
            return false;
        }
        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
        aclrtBinaryLoadOptions options{&option, 1U};
        if (!CheckAcl(
                aclrtBinaryLoadFromData(binary_data.data(), binary_data.size(), &options, &binary_),
                "aclrtBinaryLoadFromData"
            ) ||
            !CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry") ||
            !CheckAcl(
                aclrtMalloc(&device_state_, sizeof(LaunchState), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(LaunchState)"
            ) ||
            !CheckAcl(
                aclrtMalloc(&device_workspace_, kWorkloadBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(G0 workload)"
            )) {
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(device_state_) & 63U) != 0U) {
            std::fprintf(stderr, "device LaunchState is not 64-byte aligned: %p\n", device_state_);
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(device_workspace_) & 63U) != 0U) {
            std::fprintf(stderr, "device G0 workload is not 64-byte aligned: %p\n", device_workspace_);
            return false;
        }
        return true;
    }

    const std::string &SocName() const { return soc_name_; }

    uint64_t DeviceStateAddress() const { return reinterpret_cast<uint64_t>(device_state_); }

    uint64_t DeviceWorkspaceAddress() const { return reinterpret_cast<uint64_t>(device_workspace_); }

    bool Run(LaunchState *state, std::vector<float> *workspace, float *kernel_ms) {
        if (workspace == nullptr || workspace->size() * sizeof(float) != kWorkloadBytes || kernel_ms == nullptr) {
            std::fprintf(stderr, "invalid host G0 workload image\n");
            return false;
        }
        if (!CheckAcl(
                aclrtMemcpy(device_state_, sizeof(*state), state, sizeof(*state), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D LaunchState)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    device_workspace_, kWorkloadBytes, workspace->data(), kWorkloadBytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D G0 workload)"
            )) {
            return false;
        }
        struct KernelArgs {
            uint64_t state_pointer;
        } args{reinterpret_cast<uint64_t>(device_state_)};
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected mixed-kernel argument ABI");
        return CheckAcl(aclrtRecordEvent(kernel_start_event_, stream_), "aclrtRecordEvent(kernel start)") &&
               CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 32U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(mixed 1:2)"
               ) &&
               CheckAcl(aclrtRecordEvent(kernel_end_event_, stream_), "aclrtRecordEvent(kernel end)") &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(mixed kernel)") &&
               CheckAcl(
                   aclrtEventElapsedTime(kernel_ms, kernel_start_event_, kernel_end_event_),
                   "aclrtEventElapsedTime(mixed kernel)"
               ) &&
               CheckAcl(
                   aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(D2H LaunchState)"
               ) &&
               CheckAcl(
                   aclrtMemcpy(
                       workspace->data(), kWorkloadBytes, device_workspace_, kWorkloadBytes, ACL_MEMCPY_DEVICE_TO_HOST
                   ),
                   "aclrtMemcpy(D2H G0 workload)"
               );
    }

private:
    bool acl_initialized_ = false;
    bool device_set_ = false;
    int32_t device_ = 0;
    std::string soc_name_;
    aclrtStream stream_ = nullptr;
    aclrtEvent kernel_start_event_ = nullptr;
    aclrtEvent kernel_end_event_ = nullptr;
    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
    void *device_state_ = nullptr;
    void *device_workspace_ = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
    Options options{};
    if (!ParseOptions(argc, argv, &options)) {
        return EXIT_FAILURE;
    }
    std::vector<char> binary_data;
    if (!ReadBinary(options.kernel_path, &binary_data)) {
        return EXIT_FAILURE;
    }
    AclSession session;
    if (!session.Initialize(
            options.device, binary_data,
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) || defined(SIMT_CROSS_CORE_U2)
            options.acl_config.c_str()
#else
            nullptr
#endif
        )) {
        return EXIT_FAILURE;
    }
    if (session.SocName().rfind("Ascend950", 0U) != 0U) {
        std::fprintf(
            stderr, "the full-PA probe requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str()
        );
        return EXIT_FAILURE;
    }

    std::printf(
        "[DEVICE] id=%d soc=%s topology=32*(1AIC+2AIV) builders=%u "
        "simt_threads_per_builder=%u simt_threads_total=%u warps_per_builder=%u "
        "state_bytes=%zu state=0x%llx workspace_bytes=%llu workspace=0x%llx batches=%u runs=%u"
        "\n",
        options.device, session.SocName().c_str(), options.builder_count, kHostBuilderThreadCount,
        g0::BuilderThreadCount(options.builder_count), kHostBuilderWarpCount, sizeof(LaunchState),
        static_cast<unsigned long long>(session.DeviceStateAddress()), static_cast<unsigned long long>(kWorkloadBytes),
        static_cast<unsigned long long>(session.DeviceWorkspaceAddress()), options.batches, options.runs
    );

    auto state = std::make_unique<LaunchState>();
    auto initial = std::make_unique<LaunchState>();
    std::vector<float> workspace;
    std::vector<double> kernel_times_us;
    uint32_t passes = 0U;
    for (uint32_t run = 0U; run < options.runs; ++run) {
        const uint64_t nonce = UINT64_C(0xA550000000000000) ^ (static_cast<uint64_t>(options.batches) << 32U) ^
                               (static_cast<uint64_t>(options.builder_count) << 24U) ^
                               (static_cast<uint64_t>(run) + 1U);
        InitializeLaunchState(
            state.get(), nonce, options.batches, options.builder_count, session.DeviceWorkspaceAddress()
        );
        std::memcpy(initial.get(), state.get(), sizeof(*state));
        InitializeWorkspace(&workspace);
        float kernel_ms = 0.0F;
        if (!session.Run(state.get(), &workspace, &kernel_ms)) {
            std::fprintf(
                stderr, "[FAIL] run=%u nonce=0x%llx ACL launch/copy failed\n", run + 1U,
                static_cast<unsigned long long>(nonce)
            );
            break;
        }
        ValidationFailure failure{};
        if (!ValidateLaunchRun(
                *state, *initial, workspace, session.DeviceStateAddress(), nonce, options.batches,
                options.builder_count, &failure
            )) {
            const FullPaState &full_pa = FullPaView(*state);
            const FullPaState &initial_full_pa = FullPaView(*initial);
            std::fprintf(
                stderr,
                "[FAIL] run=%u nonce=0x%llx reason=%s task=%u owner=%u index=%u "
                "expected=0x%llx actual=0x%llx\n",
                run + 1U, static_cast<unsigned long long>(nonce), failure.reason, failure.task, failure.owner,
                failure.index, static_cast<unsigned long long>(failure.expected),
                static_cast<unsigned long long>(failure.actual)
            );
            if (failure.owner < g0::kOwnerCount && failure.index < g0::kTokensPerOwner &&
                std::strncmp(failure.reason, "token-control-", 14U) == 0) {
                std::array<uint64_t, sizeof(g0::ExecutionTokenControl) / sizeof(uint64_t)> actual_words{};
                std::array<uint64_t, sizeof(g0::ExecutionTokenControl) / sizeof(uint64_t)> expected_words{};
                std::memcpy(
                    actual_words.data(), &full_pa.tokens[failure.owner][failure.index].control,
                    sizeof(g0::ExecutionTokenControl)
                );
                std::memcpy(
                    expected_words.data(), &initial_full_pa.tokens[failure.owner][failure.index].control,
                    sizeof(g0::ExecutionTokenControl)
                );
                std::fprintf(
                    stderr, "[TOKEN_CONTROL] owner=%u slot=%u 8x64 actual/expected/xor:\n", failure.owner, failure.index
                );
                for (uint32_t word = 0U; word < sizeof(g0::ExecutionTokenControl) / sizeof(uint64_t); ++word) {
                    std::fprintf(
                        stderr, "  w%u=0x%016llx/0x%016llx/0x%016llx\n", word,
                        static_cast<unsigned long long>(actual_words[word]),
                        static_cast<unsigned long long>(expected_words[word]),
                        static_cast<unsigned long long>(actual_words[word] ^ expected_words[word])
                    );
                }
            }
            if (failure.owner < g0::kOwnerCount && std::strncmp(failure.reason, "role-", 5U) == 0) {
                const g0::FullPaRoleResult &role = full_pa.roles[failure.owner];
                std::fprintf(
                    stderr,
                    "[ROLE] owner=%u role=%u build=%u commit=%u execute=%u ticket=%u exhausted=%u "
                    "claim=%u lost=%u busy=%u/%u kinds=[%u,%u,%u,%u,%u] drain=%u fatal=%u\n",
                    role.owner, static_cast<uint32_t>(role.role), role.build_count, role.commit_count,
                    role.execute_count, role.ticket_count, role.exhausted_ticket_count, role.claim_count,
                    role.claim_lost_count, role.max_busy_tokens, role.final_busy_tokens, role.completed_by_kind[0],
                    role.completed_by_kind[1], role.completed_by_kind[2], role.completed_by_kind[3],
                    role.completed_by_kind[4], role.drain_arrival_count, role.fatal_count
                );
            }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            if (failure.owner < g0_swimlane::kTraceScalarWriterCount &&
                failure.index < g0_swimlane::kTraceScalarRecordsPerWriter &&
                std::strncmp(failure.reason, "swimlane-log-", 13U) == 0) {
                const g0_swimlane::TraceRecord &record = state->trace.scalar_records[failure.owner][failure.index];
                const uint64_t *record_words = reinterpret_cast<const uint64_t *>(&record);
                const g0_swimlane::TraceLogControl &control = state->trace.scalar_logs[failure.owner];
                std::fprintf(
                    stderr,
                    "[SWIMLANE_SCALAR_RECORD] owner=%u slot=%u count=%u atomic=%llu poll=%llu dcci=%llu "
                    "words=[0x%016llx,0x%016llx,0x%016llx,0x%016llx]\n",
                    failure.owner, failure.index, control.record_count,
                    static_cast<unsigned long long>(control.atomic_calls),
                    static_cast<unsigned long long>(control.poll_calls),
                    static_cast<unsigned long long>(control.dcci_calls),
                    static_cast<unsigned long long>(record_words[0]),
                    static_cast<unsigned long long>(record_words[1]),
                    static_cast<unsigned long long>(record_words[2]),
                    static_cast<unsigned long long>(record_words[3])
                );
            }
#endif
            continue;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        if (!options.swimlane_json.empty() && !WriteSwimlaneJson(*state, options.swimlane_json)) {
            std::fprintf(stderr, "[FAIL] run=%u could not export swimlane\n", run + 1U);
            continue;
        }
#endif
        ++passes;
        kernel_times_us.push_back(static_cast<double>(kernel_ms) * 1000.0);
        const FullPaState &full_pa = FullPaView(*state);
        std::array<uint32_t, g0::kMaxBuilderCount> builder_wins{};
        uint64_t insert_poll_sum = 0U;
        uint32_t insert_poll_max = 0U;
        for (uint32_t task_id = 0U; task_id < options.batches * kHostTasksPerBatch; ++task_id) {
            const uint32_t builder = full_pa.tasks[task_id].plan.builder_owner - g0::kBuilderOwner;
            ++builder_wins[builder];
            const uint32_t insert_polls = full_pa.tasks[task_id].build_report.insert_poll_count;
            insert_poll_sum += insert_polls;
            insert_poll_max = std::max(insert_poll_max, insert_polls);
        }
        std::printf(
            "[PASS] run=%u nonce=0x%llx active_tasks=%u kernel_tasks=%u heap_bytes=%llu "
            "builder_attempts_per_task=%u builder_wins=",
            run + 1U, static_cast<unsigned long long>(nonce), options.batches * kHostTasksPerBatch,
            options.batches * 4U, static_cast<unsigned long long>(ExpectedHeapBytes(options.batches)),
            1U
        );
        for (uint32_t builder = 0U; builder < options.builder_count; ++builder) {
            std::printf("%s%u", builder == 0U ? "" : "/", builder_wins[builder]);
        }
        std::printf(
            " insert_polls=%llu max_insert_polls=%u same_device_addresses=yes kernel_event_us=%.3f\n",
            static_cast<unsigned long long>(insert_poll_sum), insert_poll_max,
            static_cast<double>(kernel_ms) * 1000.0
        );
    }
    if (!kernel_times_us.empty()) {
        std::sort(kernel_times_us.begin(), kernel_times_us.end());
        long double sum = 0.0L;
        for (double value : kernel_times_us) {
            sum += value;
        }
        const size_t count = kernel_times_us.size();
        const double median = count % 2U == 0U ?
                                  (kernel_times_us[count / 2U - 1U] + kernel_times_us[count / 2U]) / 2.0 :
                                  kernel_times_us[count / 2U];
        std::printf(
            "[PERF] scope=acl_event_kernel_only trace=%s samples=%zu min_us=%.3f median_us=%.3f avg_us=%.3Lf "
            "max_us=%.3f\n",
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            "on",
#else
            "off",
#endif
            count, kernel_times_us.front(), median, sum / static_cast<long double>(count), kernel_times_us.back()
        );
    }
    std::printf(
#if defined(SIMT_CROSS_CORE_U2)
        "[SUMMARY] U2 builders=%u B%u passes=%u/%u fresh_initialization=yes same_address_reuse=%s\n",
#else
        "[SUMMARY] GM builders=%u B%u passes=%u/%u fresh_initialization=yes same_address_reuse=%s\n",
#endif
        options.builder_count, options.batches, passes, options.runs,
        options.runs > 1U ? "validated" : "not-requested"
    );
    return passes == options.runs ? EXIT_SUCCESS : EXIT_FAILURE;
}
