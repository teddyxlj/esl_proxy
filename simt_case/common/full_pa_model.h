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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_MODEL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "full_pa_exec_protocol.h"

namespace pa_scheduler::simt_cross_core::g0 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_G0_MODEL_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_G0_MODEL_INLINE constexpr
#endif

constexpr uint64_t kProbeMagic = 0x53494D5447304135ULL;
constexpr uint64_t kProbeVersion = 2U;
constexpr uint32_t kDefaultBatches = 256U;
constexpr uint32_t kTasksPerBatch = 5U;
constexpr uint32_t kKernelsPerBatch = 4U;
constexpr uint32_t kMaxTasks = 4352U;
constexpr uint32_t kMainTaskCount = kDefaultBatches * kTasksPerBatch;
constexpr uint32_t kMainKernelTaskCount = kDefaultBatches * kKernelsPerBatch;
constexpr uint32_t kAicTaskCapacity = kMaxTasks;
constexpr uint32_t kAivTaskCapacity = kMaxTasks;
constexpr uint32_t kAicOwnerCount = 32U;
constexpr uint32_t kAivOwnerCount = 64U;
constexpr uint32_t kOwnerCount = kAicOwnerCount + kAivOwnerCount;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kDefaultBuilderCount = 1U;
// 首轮多 builder 性能搜索上限，不是硬件或协议上限。8 个 builder 仍保留
// 56 个 AIV executor；若端到端最优点落在边界，再用真机数据决定是否扩展。
constexpr uint32_t kMaxBuilderCount = 8U;
constexpr uint32_t kFirstAivExecutorOwner = kBuilderOwner + 1U;
constexpr uint32_t kAivExecutorCount = kAivOwnerCount - 1U;
constexpr uint32_t kExecutorCount = kAicOwnerCount + kAivExecutorCount;
constexpr uint32_t kWarpSize = 32U;
#ifndef SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT
#define SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT 16U
#endif
// 性能扫描可由构建参数覆盖；LAUNCH_BOUND 及 host/device ABI 会随同一宏一起编译。
constexpr uint32_t kBuilderWarpCount = SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT;
constexpr uint32_t kBuilderThreadCount = kBuilderWarpCount * kWarpSize;
constexpr uint32_t kBuilderLeaderCount = kBuilderWarpCount;
constexpr uint32_t kMaxBuilderThreadCount = kBuilderThreadCount * kMaxBuilderCount;
constexpr uint32_t kMaxBuilderLeaderCount = kBuilderLeaderCount * kMaxBuilderCount;
constexpr uint32_t kBuilderTaskStride = kBuilderWarpCount;
constexpr uint32_t kBuilderTasksPerLeaderMain = kMainTaskCount / kBuilderLeaderCount;
constexpr uint32_t kSharedHeapShards = 8U;
constexpr uint64_t kHeapBytes = 256ULL << 20U;
constexpr uint64_t kHeapShardSpan = kHeapBytes / kSharedHeapShards;
constexpr uint64_t kOutputAlignment = 1024U;
constexpr uint64_t kMainReservedHeapBytes = 0x0C500000ULL;
constexpr uint64_t kMainReservedHeapBytesPerShard = 0x018A0000ULL;
constexpr uint32_t kMaxTensorDims = 5U;
constexpr uint32_t kOutputsPerTask = 8U;
constexpr uint32_t kWriterHistoryMaxPerTask = kMaxTensors;
constexpr uint32_t kWriterHistoryMagic = 0x57484953U;
constexpr uint64_t kInvalidTaskId = UINT64_MAX;
constexpr uint8_t kDispatchMetaPresent = 1U << 7U;
constexpr uint8_t kDispatchMetaLastSubmit = 1U << 6U;
constexpr uint8_t kExecRoutePresent = 1U << 0U;
constexpr uint8_t kExecRouteExecutable = 1U << 1U;
constexpr uint8_t kExecRouteEngineShift = 2U;

constexpr uint64_t kSyntheticHeapBase = 0x100000000ULL;
constexpr uint64_t kSyntheticQueryBase = 0x200000000ULL;
constexpr uint64_t kSyntheticKeyBase = 0x300000000ULL;
constexpr uint64_t kSyntheticValueBase = 0x400000000ULL;
constexpr uint64_t kSyntheticBlockTableBase = 0x500000000ULL;
constexpr uint64_t kSyntheticOutputBase = 0x600000000ULL;
constexpr uint32_t kPaHeads = 16U;
constexpr uint32_t kPaHeadDim = 128U;
constexpr uint32_t kPaBlockSize = 128U;
constexpr uint32_t kPaBlocksPerRequest = 64U;
constexpr uint32_t kPaMaxBlocksPerRequest = 256U;
constexpr uint64_t kPaScaleBits = 0x3F800000ULL;

constexpr uint32_t kWorkloadTileRows = 128U;
constexpr uint32_t kWorkloadTileColumns = 128U;
constexpr uint32_t kWorkloadTileElements = kWorkloadTileRows * kWorkloadTileColumns;
constexpr uint32_t kWorkloadTileBytes = kWorkloadTileElements * sizeof(float);
constexpr uint32_t kWorkloadSharedInputTiles = 2U;
constexpr uint32_t kWorkloadOutputTilesPerOwner = 2U;
constexpr uint32_t kWorkloadOutputTiles = kOwnerCount * kWorkloadOutputTilesPerOwner;
constexpr uint32_t kWorkloadTiles = kWorkloadSharedInputTiles + kWorkloadOutputTiles;
constexpr uint64_t kWorkloadBytes = static_cast<uint64_t>(kWorkloadTiles) * kWorkloadTileBytes;
constexpr float kWorkloadInputA = 2.0F;
constexpr float kWorkloadInputB = 3.0F;
constexpr float kWorkloadExpectedAic = 768.0F;
constexpr float kWorkloadExpectedSf = 5.0F;
constexpr float kWorkloadExpectedUp = 6.0F;
constexpr float kWorkloadOutputSentinel = -12345.0F;

enum class TaskKind : uint32_t {
    Alloc = 0,
    Qk = 1,
    Sf = 2,
    Pv = 3,
    Up = 4,
    Count = 5,
};

// 任务 schema 的 tensor 访问语义。metadata writer 的判定必须来自参数
// access tag 与 SharedOutputRef，而不是来自某个算子或 TaskKind 名称。
enum class TensorAccess : uint8_t {
    Input = 0,
    Output = 1,
    Inout = 2,
    OutputExisting = 3,
    NoDependency = 4,
};

enum class DataType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int16 = 3,
    Int8 = 4,
    Uint8 = 5,
    Bfloat16 = 6,
    Int64 = 7,
    Uint64 = 8,
    Uint16 = 9,
    Uint32 = 10,
    Bool = 11,
    Count = 12,
};

enum class OwnerRole : uint32_t {
    AicExecutor = 0,
    AivBuilder = 1,
    AivExecutor = 2,
};

struct TaskExecShape {
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
};

struct TensorDesc {
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint64_t owner_task_id;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    DataType dtype;
    bool manual_dep;
    bool is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kMaxTensorDims];
    uint64_t extent_elem_cache;
    uint32_t strides[kMaxTensorDims];
    uint8_t padding[36];
};

struct alignas(kCacheLineBytes) TaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
    volatile int64_t deps_prepared;
    uint8_t padding[kCacheLineBytes - 3U * sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) SharedOutputCell {
    AtomicLine published[kOutputsPerTask];
    AtomicLine last_writer[kOutputsPerTask];
    TensorDesc tensors[kOutputsPerTask];
};

struct WriterHistoryRecord {
    uint32_t symbol_key;
    int32_t previous_writer;
};

struct alignas(kCacheLineBytes) WriterHistoryCell {
    uint32_t magic;
    int32_t writer_task;
    uint32_t count;
    uint32_t reserved;
    WriterHistoryRecord entries[kWriterHistoryMaxPerTask];
    uint8_t padding[48];
};

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskCount(uint32_t batches) { return batches * kTasksPerBatch; }

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t KernelTaskCount(uint32_t batches) {
    return batches * kKernelsPerBatch;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskBatch(uint32_t task_id) { return task_id / kTasksPerBatch; }

SIMT_CROSS_CORE_G0_MODEL_INLINE TaskKind TaskKindAt(uint32_t task_id) {
    return static_cast<TaskKind>(task_id % kTasksPerBatch);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BatchTaskId(uint32_t batch, TaskKind kind) {
    return batch * kTasksPerBatch + static_cast<uint32_t>(kind);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool TaskExecutable(TaskKind kind) {
    return kind == TaskKind::Qk || kind == TaskKind::Sf || kind == TaskKind::Pv || kind == TaskKind::Up;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE ExecEngineClass TaskEngine(TaskKind kind) {
    return kind == TaskKind::Qk || kind == TaskKind::Pv
               ? ExecEngineClass::Aic
               : (kind == TaskKind::Sf || kind == TaskKind::Up ? ExecEngineClass::Aiv : ExecEngineClass::None);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskFunctionId(TaskKind kind) {
    return TaskExecutable(kind) ? static_cast<uint32_t>(kind) - 1U : kInvalidFunctionId;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint8_t EncodeTaskMeta(uint32_t task_id, uint32_t task_count) {
    return static_cast<uint8_t>(
        kDispatchMetaPresent | (task_id + 1U == task_count ? kDispatchMetaLastSubmit : 0U) |
        static_cast<uint32_t>(TaskKindAt(task_id))
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint8_t EncodeTaskExecRoute(TaskKind kind) {
    const ExecEngineClass engine_class = TaskEngine(kind);
    return static_cast<uint8_t>(
        kExecRoutePresent | (TaskExecutable(kind) ? kExecRouteExecutable : 0U) |
        (static_cast<uint32_t>(engine_class) << kExecRouteEngineShift)
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TaskExecShape TaskShape(TaskKind kind) {
    if (kind == TaskKind::Qk) {
        return TaskExecShape{4U, 2U, 0U, ExecEngineClass::Aic};
    }
    if (kind == TaskKind::Sf) {
        return TaskExecShape{4U, 3U, 1U, ExecEngineClass::Aiv};
    }
    if (kind == TaskKind::Pv) {
        return TaskExecShape{4U, 2U, 1U, ExecEngineClass::Aic};
    }
    if (kind == TaskKind::Up) {
        return TaskExecShape{7U, 2U, 3U, ExecEngineClass::Aiv};
    }
    return TaskExecShape{0U, 0U, 0U, ExecEngineClass::None};
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskOutputCount(TaskKind kind) {
    if (kind == TaskKind::Alloc || kind == TaskKind::Sf) {
        return 3U;
    }
    return kind == TaskKind::Qk || kind == TaskKind::Pv ? 1U : 0U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t TaskOutputBytes(TaskKind kind) {
    if (kind == TaskKind::Alloc) {
        return 10240U;
    }
    if (kind == TaskKind::Qk) {
        return 524288U;
    }
    if (kind == TaskKind::Sf) {
        return 264192U;
    }
    return kind == TaskKind::Pv ? 8192U : 0U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t AlignOutputBytes(uint64_t bytes) {
    return (bytes + kOutputAlignment - 1U) & ~(kOutputAlignment - 1U);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskHeapShard(uint32_t task_id) {
    return task_id & (kSharedHeapShards - 1U);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE int64_t InsertCompletionInitialValue(uint32_t task_id) {
    return static_cast<int64_t>(task_id) - 1;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool BuilderCountValid(uint32_t builder_count) {
    return builder_count >= kDefaultBuilderCount && builder_count <= kMaxBuilderCount;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderWarp(uint32_t thread_id) { return thread_id / kWarpSize; }

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderInstance(uint32_t thread_id) { return thread_id / kBuilderThreadCount; }

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderLocalThread(uint32_t thread_id) {
    return thread_id % kBuilderThreadCount;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderLocalWarp(uint32_t thread_id) {
    return BuilderLocalThread(thread_id) / kWarpSize;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderOwnerForInstance(uint32_t builder_instance) {
    return kBuilderOwner + builder_instance;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderOwnerForThread(uint32_t thread_id) {
    return BuilderOwnerForInstance(BuilderInstance(thread_id));
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderThreadCount(uint32_t builder_count) {
    return builder_count * kBuilderThreadCount;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t BuilderLeaderCount(uint32_t builder_count) {
    return builder_count * kBuilderLeaderCount;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool
BuilderThreadActive(uint32_t thread_id, uint32_t builder_count = kDefaultBuilderCount) {
    return BuilderCountValid(builder_count) && thread_id < BuilderThreadCount(builder_count) &&
           (thread_id % kWarpSize) == 0U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderFirstTask(uint32_t thread_id, uint32_t builder_count = kDefaultBuilderCount) {
    return BuilderThreadActive(thread_id, builder_count) ?
               BuilderInstance(thread_id) * kBuilderWarpCount + BuilderLocalWarp(thread_id) :
               UINT32_MAX;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderThreadForInstanceWarp(uint32_t builder_instance, uint32_t local_warp) {
    return builder_instance * kBuilderThreadCount + local_warp * kWarpSize;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderThreadForTask(uint32_t task_id, uint32_t builder_count = kDefaultBuilderCount) {
    const uint32_t logical_leader = task_id % BuilderLeaderCount(builder_count);
    const uint32_t builder_instance = logical_leader / kBuilderWarpCount;
    const uint32_t local_warp = logical_leader % kBuilderWarpCount;
    return BuilderThreadForInstanceWarp(builder_instance, local_warp);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderExpectedTaskCount(uint32_t thread_id, uint32_t task_count, uint32_t builder_count = kDefaultBuilderCount) {
    const uint32_t first = BuilderFirstTask(thread_id, builder_count);
    return first == UINT32_MAX || first >= task_count ?
               0U :
               1U + (task_count - 1U - first) / BuilderLeaderCount(builder_count);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderExpectedLastTask(uint32_t thread_id, uint32_t task_count, uint32_t builder_count = kDefaultBuilderCount) {
    const uint32_t count = BuilderExpectedTaskCount(thread_id, task_count, builder_count);
    return count == 0U ?
               UINT32_MAX :
               BuilderFirstTask(thread_id, builder_count) + (count - 1U) * BuilderLeaderCount(builder_count);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t
BuilderExpectedInsertWaitCount(uint32_t thread_id, uint32_t task_count, uint32_t builder_count = kDefaultBuilderCount) {
    const uint32_t count = BuilderExpectedTaskCount(thread_id, task_count, builder_count);
    return count - (count != 0U && BuilderFirstTask(thread_id, builder_count) == 0U ? 1U : 0U);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t BuilderReportChecksum(
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
    value *= 0x9E3779B97F4A7C15ULL;
    value ^= value >> 29U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t
BuilderReportChecksum(uint64_t nonce, uint32_t thread_id, uint32_t task_count) {
    const uint32_t wins = BuilderExpectedTaskCount(thread_id, task_count);
    return BuilderReportChecksum(
        nonce, thread_id, task_count, wins, wins == 0U ? UINT32_MAX : BuilderFirstTask(thread_id),
        BuilderExpectedLastTask(thread_id, task_count), wins, wins, wins,
        BuilderExpectedInsertWaitCount(thread_id, task_count), 0U
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool
ConsecutiveTasksHaveSafeBuilderMapping(uint32_t task_id, uint32_t builder_count = kDefaultBuilderCount) {
    if (task_id == 0U) {
        return true;
    }
    return BuilderWarp(BuilderThreadForTask(task_id, builder_count)) !=
           BuilderWarp(BuilderThreadForTask(task_id - 1U, builder_count));
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool IsBuilderOwner(uint32_t owner, uint32_t builder_count = kDefaultBuilderCount) {
    return BuilderCountValid(builder_count) && owner >= kBuilderOwner && owner < kBuilderOwner + builder_count;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t AivExecutorCount(uint32_t builder_count = kDefaultBuilderCount) {
    return BuilderCountValid(builder_count) ? kAivOwnerCount - builder_count : 0U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t ExecutorCount(uint32_t builder_count = kDefaultBuilderCount) {
    return kAicOwnerCount + AivExecutorCount(builder_count);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE OwnerRole OwnerRoleAt(uint32_t owner, uint32_t builder_count = kDefaultBuilderCount) {
    return owner < kAicOwnerCount ?
               OwnerRole::AicExecutor :
               (IsBuilderOwner(owner, builder_count) ? OwnerRole::AivBuilder : OwnerRole::AivExecutor);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE ExecEngineClass
OwnerEngine(uint32_t owner, uint32_t builder_count = kDefaultBuilderCount) {
    return owner < kAicOwnerCount ?
               ExecEngineClass::Aic :
               (owner < kOwnerCount && !IsBuilderOwner(owner, builder_count) ? ExecEngineClass::Aiv :
                                                                               ExecEngineClass::None);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE bool
OwnerCanExecute(uint32_t owner, ExecEngineClass engine_class, uint32_t builder_count = kDefaultBuilderCount) {
    return owner < kOwnerCount && !IsBuilderOwner(owner, builder_count) &&
           OwnerEngine(owner, builder_count) == engine_class;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t OwnerPhysicalBlock(uint32_t owner) {
    return owner < kAicOwnerCount ? owner : (owner - kAicOwnerCount) / 2U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t OwnerDrainGroup(uint32_t owner) {
    return OwnerPhysicalBlock(owner) % kDrainGroupCount;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t AicDispatchTaskId(uint32_t index) {
    const uint32_t batch = index / 2U;
    return BatchTaskId(batch, (index & 1U) == 0U ? TaskKind::Qk : TaskKind::Pv);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t AivDispatchTaskId(uint32_t index) {
    const uint32_t batch = index / 2U;
    return BatchTaskId(batch, (index & 1U) == 0U ? TaskKind::Sf : TaskKind::Up);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskScalarCount(uint32_t task_id) {
    return TaskShape(TaskKindAt(task_id)).scalar_count;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t TaskScalar(uint32_t task_id, uint32_t scalar_index) {
    const TaskKind kind = TaskKindAt(task_id);
    const uint32_t batch = TaskBatch(task_id);
    if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
        return scalar_index == 0U ? kPaBlocksPerRequest : static_cast<uint64_t>(batch) * kPaMaxBlocksPerRequest;
    }
    if (kind == TaskKind::Sf) {
        return scalar_index == 0U ? kPaScaleBits : (scalar_index == 1U ? kPaBlocksPerRequest : kPaBlockSize);
    }
    if (kind == TaskKind::Up) {
        return 1U;
    }
    return 0U;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t TaskFaninCount(uint32_t task_id) {
    return TaskShape(TaskKindAt(task_id)).fanin_count;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE int32_t TaskFanin(uint32_t task_id, uint32_t edge) {
    const TaskKind kind = TaskKindAt(task_id);
    if (kind == TaskKind::Sf || kind == TaskKind::Pv) {
        return edge == 0U ? static_cast<int32_t>(task_id - 1U) : -1;
    }
    if (kind == TaskKind::Up) {
        if (edge == 0U) {
            return static_cast<int32_t>(task_id - 2U);
        }
        if (edge == 1U) {
            return static_cast<int32_t>(task_id - 1U);
        }
        return edge == 2U ? static_cast<int32_t>(BatchTaskId(TaskBatch(task_id), TaskKind::Alloc)) : -1;
    }
    return -1;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t DataTypeBytes(DataType dtype) {
    return dtype == DataType::Float32 || dtype == DataType::Int32 || dtype == DataType::Uint32
               ? 4U
               : (dtype == DataType::Float16 || dtype == DataType::Int16 || dtype == DataType::Bfloat16 ||
                          dtype == DataType::Uint16
                      ? 2U
                      : (dtype == DataType::Int64 || dtype == DataType::Uint64 ? 8U : 1U));
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeContiguousTensorDesc(
    uint64_t address, uint64_t owner_task_id, uint64_t start_offset, uint32_t ndims, DataType dtype,
    bool manual_dep, uint32_t shape0, uint32_t shape1
) {
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.owner_task_id = owner_task_id;
    tensor.start_offset = start_offset;
    tensor.version = 0;
    tensor.ndims = ndims;
    tensor.dtype = dtype;
    tensor.manual_dep = manual_dep;
    tensor.is_contiguous = true;
    tensor.child_memory = 0U;
    tensor.shapes[0] = shape0;
    tensor.shapes[1] = shape1;
    tensor.strides[0] = ndims == 1U ? 1U : shape1;
    tensor.strides[1] = ndims == 1U ? 0U : 1U;
    tensor.extent_elem_cache = ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    tensor.buffer_size = tensor.extent_elem_cache * DataTypeBytes(dtype);
    return tensor;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeQueryViewDescriptor(uint32_t batches, uint32_t batch) {
    TensorDesc tensor = MakeContiguousTensorDesc(
        kSyntheticQueryBase, kInvalidTaskId, static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim, 2U,
        DataType::Bfloat16, false, kPaHeads, kPaHeadDim
    );
    tensor.buffer_size = static_cast<uint64_t>(batches) * kPaHeads * kPaHeadDim * DataTypeBytes(DataType::Bfloat16);
    return tensor;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeKeyOrValueDescriptor(uint32_t batches, bool value) {
    return MakeContiguousTensorDesc(
        value ? kSyntheticValueBase : kSyntheticKeyBase, kInvalidTaskId, 0U, 2U, DataType::Bfloat16, false,
        batches * kPaBlocksPerRequest * kPaBlockSize, kPaHeadDim
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeBlockTableDescriptor(uint32_t batches) {
    return MakeContiguousTensorDesc(
        kSyntheticBlockTableBase, kInvalidTaskId, 0U, 2U, DataType::Int32, false, batches,
        kPaMaxBlocksPerRequest
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeOutputViewDescriptor(uint32_t batches, uint32_t batch) {
    TensorDesc tensor = MakeContiguousTensorDesc(
        kSyntheticOutputBase, kInvalidTaskId, static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim, 2U,
        DataType::Float32, true, kPaHeads, kPaHeadDim
    );
    tensor.buffer_size = static_cast<uint64_t>(batches) * kPaHeads * kPaHeadDim * DataTypeBytes(DataType::Float32);
    return tensor;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE TensorDesc MakeTaskOutputDescriptor(
    uint32_t task_id, uint32_t output_slot, uint64_t task_heap_base
) {
    const TaskKind kind = TaskKindAt(task_id);
    uint64_t output_offset = 0U;
    uint32_t ndims = 0U;
    DataType dtype = DataType::Float32;
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == TaskKind::Alloc) {
        ndims = output_slot == 0U ? 2U : 1U;
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaHeadDim : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 8192U : 9216U);
    } else if (kind == TaskKind::Qk) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaBlocksPerRequest * kPaBlockSize;
    } else if (kind == TaskKind::Sf) {
        ndims = output_slot == 0U ? 2U : 1U;
        dtype = output_slot == 0U ? DataType::Bfloat16 : DataType::Float32;
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaBlocksPerRequest * kPaBlockSize : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 262144U : 263168U);
    } else if (kind == TaskKind::Pv) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    return MakeContiguousTensorDesc(
        kSyntheticHeapBase + task_heap_base + output_offset, task_id, 0U, ndims, dtype, false, shape0, shape1
    );
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t WriterHistorySymbolKey(uint32_t batch, uint32_t history_index) {
    return BatchTaskId(batch, TaskKind::Alloc) * kOutputsPerTask + 3U - history_index;
}

SIMT_CROSS_CORE_G0_MODEL_INLINE float ExpectedWorkloadValue(TaskKind kind) {
    return kind == TaskKind::Qk || kind == TaskKind::Pv
               ? kWorkloadExpectedAic
               : (kind == TaskKind::Sf ? kWorkloadExpectedSf : kWorkloadExpectedUp);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint32_t ExpectedWorkloadValueBits(TaskKind kind) {
    return kind == TaskKind::Qk || kind == TaskKind::Pv
               ? 0x44400000U
               : (kind == TaskKind::Sf ? 0x40A00000U : 0x40C00000U);
}

SIMT_CROSS_CORE_G0_MODEL_INLINE uint64_t ExpectedWorkloadOutputPair(TaskKind kind) {
    const uint64_t bits = ExpectedWorkloadValueBits(kind);
    return bits | (bits << 32U);
}

static_assert(kMainTaskCount == 1280U && kMainKernelTaskCount == 1024U, "main PA task counts changed");
static_assert(
    kBuilderWarpCount >= 1U && kBuilderWarpCount <= 64U && kBuilderThreadCount == kBuilderWarpCount * kWarpSize &&
        kBuilderLeaderCount == kBuilderWarpCount,
    "GM builder warp count must describe a 1..64 warp, lane0-only launch"
);
static_assert(
    kMaxBuilderThreadCount == kBuilderThreadCount * kMaxBuilderCount &&
        kMaxBuilderLeaderCount == kBuilderLeaderCount * kMaxBuilderCount,
    "GM builder scaling must retain up to eight independent configured-warp instances"
);
static_assert(
    kOwnerCount == 96U && kExecutorCount == 95U && kAicOwnerCount + kAivOwnerCount - kMaxBuilderCount == 88U,
    "mixed owner topology changed"
);
static_assert(kWorkloadTileBytes == 65536U && kWorkloadBytes == 12713984U, "workload layout changed");
static_assert(sizeof(TensorDesc) == kTensorDescBytes, "TensorDesc ABI changed");
static_assert(offsetof(TensorDesc, shapes) == 44U && offsetof(TensorDesc, extent_elem_cache) == 64U &&
                  offsetof(TensorDesc, strides) == 72U,
              "TensorDesc field offsets changed");
static_assert(sizeof(TaskCell) == kCacheLineBytes, "TaskCell ABI changed");
static_assert(sizeof(SharedOutputCell) == 2048U, "SharedOutputCell ABI changed");
static_assert(offsetof(SharedOutputCell, last_writer) == 512U && offsetof(SharedOutputCell, tensors) == 1024U,
              "SharedOutputCell offsets changed");
static_assert(sizeof(WriterHistoryCell) == 320U, "WriterHistoryCell ABI changed");
static_assert(offsetof(WriterHistoryCell, entries) == 16U, "writer history entries offset changed");
static_assert(10240U + 524288U + 264192U + 8192U == 806912U,
              "per-batch shared heap footprint changed");

#undef SIMT_CROSS_CORE_G0_MODEL_INLINE

}  // namespace pa_scheduler::simt_cross_core::g0

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_FULL_PA_MODEL_H
