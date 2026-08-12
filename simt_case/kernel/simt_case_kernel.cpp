#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>
#include "cce_aicore_intrinsics.h"

#include "../common/simt_case_protocol.h"
#include "../common/g0_full_pa.h"
#include "full_pa_workloads.h"

#ifndef SIMT_CASE_WARP_COUNT
#define SIMT_CASE_WARP_COUNT 8
#endif

namespace {

using namespace pa_scheduler::simt_cross_core::g0;
using namespace pa_scheduler::simt_cross_core::g0::device;
using namespace pto;

constexpr uint32_t kSimtWarpSize = 32;
constexpr uint32_t kSimtWarpCount = SIMT_CASE_WARP_COUNT;
constexpr uint32_t kSimtThreadCount = kSimtWarpCount * kSimtWarpSize;
constexpr int kSingleCacheLine = 0;
constexpr uint32_t kWatchdogMask = 0x3FFU;
constexpr uint32_t kDrainExpectedArrivals = 6U;

__aicore__ inline uint64_t ScalarAtomicLoad(__gm__ volatile int64_t *addr) {
    return (uint64_t)atomicAdd((__gm__ int64_t *)addr, (int64_t)0);
}

__aicore__ inline uint64_t ScalarCAS(__gm__ volatile int64_t *addr, uint64_t expected, uint64_t desired) {
    return (uint64_t)atomicCAS((__gm__ int64_t *)addr, (int64_t)expected, (int64_t)desired);
}

__aicore__ inline uint64_t ScalarFetchAdd(__gm__ volatile int64_t *addr, int64_t inc) {
    return (uint64_t)atomicAdd((__gm__ int64_t *)addr, inc);
}

__aicore__ inline uint64_t LoadDev64(__gm__ const uint64_t *addr) {
    return (uint64_t)__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(addr), 0);
}

__aicore__ inline void StoreDev64(__gm__ uint64_t *addr, uint64_t value) {
    __builtin_cce_st_dev(value, addr, 0);
}

__aicore__ inline uint64_t ScalarExchange(__gm__ volatile int64_t *addr, uint64_t value) {
    return (uint64_t)atomicExch((__gm__ int64_t *)addr, (int64_t)value);
}

__aicore__ inline uint64_t LoadFatal(__gm__ FullPaState *state) {
    return ScalarAtomicLoad(&state->fatal.state);
}

__aicore__ inline void
PublishFatal(__gm__ FullPaState *state, ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)ScalarCAS(&state->fatal.state, 0U, EncodeExecFatal(reason, owner, task_id));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const FullPaState *state) {
    const uint32_t batches = state->control.batch_count;
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.timeout_ticks != 0U && batches >= 1U && batches <= kDefaultBatches &&
           state->control.task_count == TaskCount(batches) &&
           state->control.kernel_task_count == KernelTaskCount(batches) &&
           state->control.builder_thread_count == kBuilderThreadCount &&
           BuilderCountValid(state->control.builder_count) && state->control.heap_base == kSyntheticHeapBase &&
           state->control.heap_bytes == kHeapBytes && state->control.workspace_base != 0U &&
           state->control.workspace_bytes == kWorkloadBytes && state->control.qk_repeats >= 1U &&
           state->control.sf_repeats >= 1U && state->control.pv_repeats >= 1U && state->control.up_repeats >= 1U &&
           state->exec_dispatch.aic_task_count == batches * 2U && state->exec_dispatch.aiv_task_count == batches * 2U;
}

__aicore__ __attribute__((always_inline)) inline void ResetToken(__gm__ ExecutionToken *token) {
    token->control.phase = ExecTokenPhase::Idle;
    token->control.task_id = UINT32_MAX;
    token->control.build_owner = UINT32_MAX;
    token->control.execute_owner = UINT32_MAX;
    token->control.engine_class = ExecEngineClass::None;
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.fanin_ready_prefix = 0U;
    token->control.payload_address = 0U;
    token->control.completion_vend = 0U;
    token->control.function_and_reference = 0U;
    token->control.shape_and_scalar_offset = 0U;
}

__aicore__ __attribute__((always_inline)) inline uint32_t
BusyTokenCount(__gm__ FullPaState *state, uint32_t owner) {
    uint32_t busy = 0U;
    for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
        if (state->tokens[owner][slot].control.phase != ExecTokenPhase::Idle) {
            ++busy;
        }
    }
    return busy;
}

__aicore__ __attribute__((always_inline)) inline void
PublishTerminalTokenState(__gm__ FullPaState *state, uint32_t owner, uint32_t ticket_count) {
    const uint32_t used_tokens = ticket_count < kTokensPerOwner ? ticket_count : kTokensPerOwner;
    for (uint32_t slot = 0U; slot < used_tokens; ++slot) {
        __gm__ ExecutionToken *token = &state->tokens[owner][slot];
        dcci(static_cast<__gm__ void *>(&token->control), kSingleCacheLine);
        __gm__ uint8_t *dispatch = reinterpret_cast<__gm__ uint8_t *>(&token->dispatch);
        dcci(static_cast<__gm__ void *>(dispatch), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 6U * kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 7U * kCacheLineBytes), kSingleCacheLine);
    }
    if (used_tokens != 0U) {
        dsb(DSB_ALL);
    }
}

__aicore__ __attribute__((always_inline)) inline uint32_t
LoadDispatchTaskId(__gm__ const uint32_t *task_ids, uint32_t ticket) {
    __gm__ const uint32_t *line = task_ids + (ticket & ~15U);
    dcci(static_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(line)), kSingleCacheLine);
    dsb(DSB_ALL);
    return task_ids[ticket];
}

__aicore__ __attribute__((always_inline)) inline bool
ValidatePayloadAndBind(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token) {
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    dcci(static_cast<__gm__ void *>(&task->exec.payload), kSingleCacheLine);
    dsb(DSB_ALL);
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.completion_vend = kCompletionSequenceWorkloadWitnessVendFlagDone;
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
FaninReady(__gm__ FullPaState *state, __gm__ ExecutionToken *token) {
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
PublishExecutionWitness(
    __gm__ FullPaState *state, uint32_t owner, uint32_t task_id, TaskKind kind,
    uint64_t checksum, uint64_t fanin_ready_prefix
) {
    __gm__ FullPaTask *task = &state->tasks[task_id];
    __gm__ FullPaExecutionWitness *witness = &task->execution_witness;
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(witness);
    StoreDev64(words + 0U, ExecutionWitnessState(state->control.launch_nonce, task_id, kind, owner));
    StoreDev64(words + 1U, state->control.launch_nonce);
    StoreDev64(words + 2U, kExecutionWitnessMagic);
    StoreDev64(words + 3U, static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(kind) << 32U));
    StoreDev64(words + 4U, static_cast<uint64_t>(owner));
    StoreDev64(words + 5U, checksum);
    StoreDev64(words + 6U, fanin_ready_prefix);
    StoreDev64(words + 7U, kCompletionSequenceWorkloadWitnessVendFlagDone);
    dsb(DSB_ALL);
    return ScalarCAS(&witness->state, 0U,
        ExecutionWitnessState(state->control.launch_nonce, task_id, kind, owner)) == 0U;
}

__aicore__ __attribute__((always_inline)) inline bool
RunClaimedWorkload(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token) {
    const uint32_t task_id = token->control.task_id;
    const TaskKind kind = TaskKindAt(task_id);
    __gm__ float *workspace = reinterpret_cast<__gm__ float *>(state->control.workspace_base);
    __gm__ float *input_a = workspace;
    __gm__ float *input_b = workspace + kWorkloadTileElements;
    const uint32_t kind_slot = kind == TaskKind::Pv || kind == TaskKind::Up ? 1U : 0U;
    __gm__ float *output = workspace + (kWorkloadSharedInputTiles + owner * kWorkloadOutputTilesPerOwner + kind_slot) *
                                           kWorkloadTileElements;
    uint64_t output_poison = state->control.launch_nonce ^
                             (static_cast<uint64_t>(task_id) * UINT64_C(0x9E3779B97F4A7C15)) ^
                             UINT64_C(0xD15EA5E0C001D00D);
    if (output_poison == ExpectedWorkloadOutputPair(kind)) {
        output_poison ^= UINT64_C(0xFFFFFFFFFFFFFFFF);
    }
    StoreDev64(reinterpret_cast<__gm__ uint64_t *>(output), output_poison);
    dsb(DSB_ALL);
#if defined(__DAV_VEC__)
    if (kind == TaskKind::Sf) {
        RunG0VectorAdd(input_a, input_b, output, state->control.sf_repeats);
    } else if (kind == TaskKind::Up) {
        RunG0VectorMultiply(input_a, input_b, output, state->control.up_repeats);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
#else
    if (kind == TaskKind::Qk) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.qk_repeats);
    } else if (kind == TaskKind::Pv) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.pv_repeats);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
#endif
    const uint64_t checksum = LoadDev64(reinterpret_cast<__gm__ const uint64_t *>(output));
    if (checksum != ExpectedWorkloadOutputPair(kind)) {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
    if (!PublishExecutionWitness(
            state, owner, task_id, kind, checksum,
            token->control.fanin_ready_prefix
        )) {
        PublishFatal(state, ExecFatalReason::CompletionPublishFailed, owner, task_id);
        return false;
    }
    __gm__ FullPaTask *task = &state->tasks[task_id];
    (void)ScalarExchange(
        reinterpret_cast<__gm__ volatile int64_t *>(const_cast<__gm__ uint64_t *>(&task->completion.vend)),
        token->control.completion_vend);
    (void)ScalarExchange(&task->completion.flag, 1U);
    const uint64_t claimed = ClaimedState(task_id, token->control.build_owner, owner);
    (void)ScalarCAS(&task->exec.control.state, claimed, DoneState(task_id, token->control.build_owner, owner));
    (void)ScalarFetchAdd(&state->drain.done_count.value, 1U);
    if (TaskEngine(kind) == ExecEngineClass::Aic) {
        (void)ScalarFetchAdd(&state->drain.aic_done.value, 1U);
    } else {
        (void)ScalarFetchAdd(&state->drain.aiv_done.value, 1U);
    }
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
AdvanceToken(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token, FullPaRoleResult *result) {
    if (token->control.phase == ExecTokenPhase::Idle) {
        return false;
    }
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    if (token->control.phase == ExecTokenPhase::WaitingBuilt) {
        const uint64_t observed = ScalarAtomicLoad(&task->exec.control.state);
        if (observed == 0U) {
            return false;
        }
        const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(observed));
        if (decoded.valid && decoded.phase == ExecPhase::Building && decoded.task_id == task_id &&
            IsBuilderOwner(decoded.build_owner, state->control.builder_count) &&
            observed == BuildingState(task_id, decoded.build_owner)) {
            return false;
        }
        if (!decoded.valid || decoded.phase != ExecPhase::Built || decoded.task_id != task_id ||
            !IsBuilderOwner(decoded.build_owner, state->control.builder_count) ||
            observed != BuiltState(task_id, decoded.build_owner)) {
            ++result->claim_lost_count;
            PublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner, task_id);
            return false;
        }
        token->control.build_owner = decoded.build_owner;
        const uint64_t claimed = ClaimedState(task_id, decoded.build_owner, owner);
        if (ScalarCAS(&task->exec.control.state, observed, claimed) != observed) {
            ++result->claim_lost_count;
            PublishFatal(state, ExecFatalReason::ControlPublishConflict, owner, task_id);
            return false;
        }
        ++result->claim_count;
        token->control.phase = ExecTokenPhase::Binding;
        if (!ValidatePayloadAndBind(state, owner, token)) {
            PublishFatal(state, ExecFatalReason::ClaimedPayloadInvalid, owner, task_id);
            return false;
        }
        token->control.phase = ExecTokenPhase::WaitingFanin;
    }
    if (token->control.phase == ExecTokenPhase::WaitingFanin) {
        if (!FaninReady(state, token)) {
            return false;
        }
        token->control.phase = ExecTokenPhase::EngineInflight;
    }
    if (token->control.phase == ExecTokenPhase::EngineInflight) {
        token->control.phase = ExecTokenPhase::Completing;
        if (!RunClaimedWorkload(state, owner, token)) {
            return false;
        }
        ++result->execute_count;
        ++result->completed_by_kind[static_cast<uint32_t>(TaskKindAt(task_id))];
        ResetToken(token);
        return true;
    }
    return false;
}

__aicore__ __attribute__((always_inline)) inline void
RunExecutor(__gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result) {
    const ExecEngineClass engine = OwnerEngine(owner, state->control.builder_count);
    __gm__ AtomicLine *cursor =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_next : &state->exec_dispatch.aiv_next;
    __gm__ uint32_t *task_ids =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_task_ids[0] : &state->exec_dispatch.aiv_task_ids[0];
    const uint32_t task_count =
        engine == ExecEngineClass::Aic ? state->exec_dispatch.aic_task_count : state->exec_dispatch.aiv_task_count;
    bool exhausted = false;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t iterations = 0U;
    while (true) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner && !exhausted; ++slot) {
            __gm__ ExecutionToken *token = &state->tokens[owner][slot];
            if (token->control.phase != ExecTokenPhase::Idle) {
                continue;
            }
            const uint32_t ticket = static_cast<uint32_t>(ScalarFetchAdd(&cursor->value, 1U));
            if (ticket >= task_count) {
                exhausted = true;
                ++result->exhausted_ticket_count;
                break;
            }
            const uint32_t task_id = LoadDispatchTaskId(task_ids, ticket);
            token->control.phase = ExecTokenPhase::WaitingBuilt;
            token->control.task_id = task_id;
            token->control.build_owner = UINT32_MAX;
            token->control.execute_owner = owner;
            token->control.engine_class = engine;
            token->control.payload_address = reinterpret_cast<uint64_t>(&state->tasks[task_id].exec.payload);
            ++result->ticket_count;
            const uint32_t busy = BusyTokenCount(state, owner);
            result->max_busy_tokens = busy > result->max_busy_tokens ? busy : result->max_busy_tokens;
        }
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            (void)AdvanceToken(state, owner, &state->tokens[owner][slot], result);
        }
        if (exhausted && BusyTokenCount(state, owner) == 0U) {
            break;
        }
        ++iterations;
        if ((iterations & kWatchdogMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                break;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX);
                break;
            }
        }
    }
    if (LoadFatal(state) != 0U) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            ResetToken(&state->tokens[owner][slot]);
        }
    }
    PublishTerminalTokenState(state, owner, result->ticket_count);
    result->final_busy_tokens = BusyTokenCount(state, owner);
}

__aicore__ __attribute__((always_inline)) inline void
PublishRoleResult(__gm__ FullPaState *state, uint32_t owner, const FullPaRoleResult &result) {
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&state->roles[owner]);
    StoreDev64(destination + 0U, static_cast<uint64_t>(result.owner) | (static_cast<uint64_t>(result.role) << 32U));
    StoreDev64(destination + 1U, static_cast<uint64_t>(result.physical_block) | (static_cast<uint64_t>(result.drain_group) << 32U));
    StoreDev64(destination + 2U, static_cast<uint64_t>(result.build_count) | (static_cast<uint64_t>(result.commit_count) << 32U));
    StoreDev64(destination + 3U, static_cast<uint64_t>(result.execute_count) | (static_cast<uint64_t>(result.ticket_count) << 32U));
    StoreDev64(destination + 4U, static_cast<uint64_t>(result.exhausted_ticket_count) | (static_cast<uint64_t>(result.claim_count) << 32U));
    StoreDev64(destination + 5U, static_cast<uint64_t>(result.claim_lost_count) | (static_cast<uint64_t>(result.max_busy_tokens) << 32U));
    StoreDev64(destination + 6U, static_cast<uint64_t>(result.final_busy_tokens) | (static_cast<uint64_t>(result.completed_by_kind[0]) << 32U));
    StoreDev64(destination + 7U, static_cast<uint64_t>(result.completed_by_kind[1]) | (static_cast<uint64_t>(result.completed_by_kind[2]) << 32U));
    StoreDev64(destination + 8U, static_cast<uint64_t>(result.completed_by_kind[3]) | (static_cast<uint64_t>(result.completed_by_kind[4]) << 32U));
    StoreDev64(destination + 9U, static_cast<uint64_t>(result.drain_arrival_count) | (static_cast<uint64_t>(result.fatal_count) << 32U));
    StoreDev64(destination + 10U, result.launch_nonce);
    StoreDev64(destination + 11U, result.reserved[0]);
    StoreDev64(destination + 12U, result.reserved[1]);
    StoreDev64(destination + 13U, result.reserved[2]);
    StoreDev64(destination + 14U, result.reserved[3]);
    StoreDev64(destination + 15U, result.reserved[4]);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline void
InitializeRoleResult(FullPaRoleResult *result, uint32_t owner, uint32_t builder_count, uint64_t nonce) {
    result->owner = owner;
    result->role = OwnerRoleAt(owner, builder_count);
    result->physical_block = OwnerPhysicalBlock(owner);
    result->drain_group = OwnerDrainGroup(owner);
    result->build_count = 0U;
    result->commit_count = 0U;
    result->execute_count = 0U;
    result->ticket_count = 0U;
    result->exhausted_ticket_count = 0U;
    result->claim_count = 0U;
    result->claim_lost_count = 0U;
    result->max_busy_tokens = 0U;
    result->final_busy_tokens = 0U;
    result->completed_by_kind[0] = 0U;
    result->completed_by_kind[1] = 0U;
    result->completed_by_kind[2] = 0U;
    result->completed_by_kind[3] = 0U;
    result->completed_by_kind[4] = 0U;
    result->drain_arrival_count = 0U;
    result->fatal_count = 0U;
    result->launch_nonce = nonce;
    result->reserved[0] = 0U;
    result->reserved[1] = 0U;
    result->reserved[2] = 0U;
    result->reserved[3] = 0U;
    result->reserved[4] = 0U;
}

__aicore__ __attribute__((always_inline)) inline void
ArriveAndDrain(__gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result) {
    result->drain_arrival_count = 1U;
    result->fatal_count = LoadFatal(state) == 0U ? 0U : 1U;
    PublishRoleResult(state, owner, *result);
    const int64_t contribution = EncodeDrainContribution(result->execute_count);
    (void)ScalarFetchAdd(&state->drain.arrivals[result->drain_group].value, contribution);
    if (owner != kBuilderOwner) {
        return;
    }
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t completed = 0U;
    while (true) {
        bool all_arrived = true;
        completed = 0U;
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
            const int64_t raw = static_cast<int64_t>(ScalarAtomicLoad(&state->drain.arrivals[group].value));
            all_arrived = all_arrived && DecodeDrainArrivals(raw) == kDrainExpectedArrivals;
            completed += DecodeDrainCompletions(raw);
        }
        if (all_arrived) {
            break;
        }
        if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
            PublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX);
            break;
        }
    }
    if (LoadFatal(state) == 0U) {
        StoreDev64(reinterpret_cast<__gm__ uint64_t *>(&state->drain.root_finished), 1U);
        dsb(DSB_ALL);
    }
}

}

#if defined(__DAV_VEC__)

using namespace pa_scheduler::simt_cross_core::g0;
using namespace pa_scheduler::simt_cross_core::g0::device;

extern "C" __simd_vf__ __aicore__ void simt_case_simd_anchor(__ubuf__ uint32_t *scratch) {
    scratch[0] = scratch[0] + 1U;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kSimtThreadCount) void
SimtDescVF(
    __gm__ uint64_t *state_words,
    uint64_t total_task_cnt,
    uint64_t warp_count
) {
    const uint32_t thread = (uint32_t)threadIdx.x;
    const uint32_t warp = thread / kSimtWarpSize;
    const uint32_t lane = thread % kSimtWarpSize;

    if (warp >= (uint32_t)warp_count) return;

    __gm__ FullPaState *state = (__gm__ FullPaState *)state_words;

    for (uint32_t task_id = warp; task_id < (uint32_t)total_task_cnt; task_id += (uint32_t)warp_count) {
        __gm__ FullPaTask *task = &state->tasks[task_id];
        task->plan.task_id = task_id;
        task->plan.batch = task_id / 5U;
        task->plan.kind = static_cast<TaskKind>(task_id % 5U);
        task->exec.control.state = static_cast<int64_t>(1);
        task->completion.flag = 0;
        task->completion.vend = 0;
        task->execution_witness.state = 0;
        if (lane == 0) {
            asc_atomic_add((__gm__ uint64_t *)&state->drain.builder_finished.value, 1);
        }
    }
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aiv(__gm__ FullPaState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes), kSingleCacheLine);
    dcci(static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes), kSingleCacheLine);
    dsb(DSB_ALL);

    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock_dim = static_cast<uint32_t>(get_subblockdim());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    const uint32_t aiv_id = block * subblock_dim + subblock;
    const uint32_t owner = kBuilderOwner + aiv_id;

    if (aiv_id == 0) {
        cce::async_invoke<SimtDescVF>(
            cce::dim3{kSimtThreadCount, 1U, 1U},
            (__gm__ uint64_t *)state,
            (uint64_t)state->control.task_count,
            (uint64_t)kSimtWarpCount
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }

    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);

    if (IsBuilderOwner(owner, state->control.builder_count)) {
        ArriveAndDrain(state, owner, &result);
        return;
    }

    if (ConfigValid(state)) {
        RunExecutor(state, owner, &result);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U);
    }
    ArriveAndDrain(state, owner, &result);
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_case_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_case_aic(__gm__ FullPaState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes), kSingleCacheLine);
    dcci(static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes), kSingleCacheLine);
    dsb(DSB_ALL);

    const uint32_t owner = static_cast<uint32_t>(get_block_idx());
    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);

    if (ConfigValid(state)) {
        RunExecutor(state, owner, &result);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U);
    }
    ArriveAndDrain(state, owner, &result);
}

#endif
