/*
 * simt_case_aicpu.cpp — AICPU kernel (runs on A5 ARM CPU)
 *
 * Phase 1 (alloc): build Qwen3 decode task graph in FullPaState
 *   - Fill tasks[].plan + exec.payload + BuiltState
 *   - Fill exec_dispatch task_ids
 *   - Set drain counters
 *
 * The reference AICore kernel's executor then picks up tasks and executes them.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>

/* Make g0_full_pa.h compile on AICPU (no __gm__ / __aicore__) */
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__
#endif

#include "../common/g0_full_pa.h"
#include "../common/full_pa_model.h"

using namespace pa_scheduler::simt_cross_core::g0;

/* ── Qwen3 task graph builder ── */

namespace {

constexpr uint32_t kQ3Tiles = 6;
constexpr uint32_t kQ3Rows = 90;

struct Q3Task {
    TaskKind kind;
    ExecEngineClass engine;
};

/* Build a simplified Qwen3-like task graph using PA task kinds.
 * We use 256 batches × 5 tasks = 1280 tasks = 1024 kernel tasks.
 * Each batch: Alloc(0), Qk(1), Sf(2), Pv(3), Up(4)
 * Qk/Pv → AIC executor (Cube matmul)
 * Sf/Up → AIV executor (Vector add/multiply)
 * Alloc → skipped (not a kernel task)
 */

void init_state(FullPaState *state, uint64_t nonce, uint32_t batches, uint64_t workspace_addr) {
    /* only zero the control region — skip full 32MB memset */
    std::memset(&state->control, 0, sizeof(state->control));
    std::memset(&state->fatal, 0, sizeof(state->fatal));
    std::memset(&state->drain, 0, sizeof(state->drain));
    std::memset(&state->exec_dispatch, 0, sizeof(state->exec_dispatch));

    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = 30000000000ULL;
    state->control.batch_count = batches;
    state->control.task_count = TaskCount(batches);
    state->control.kernel_task_count = KernelTaskCount(batches);
    state->control.builder_thread_count = kBuilderThreadCount;
    state->control.heap_base = kSyntheticHeapBase;
    state->control.heap_bytes = kHeapBytes;
    state->control.workspace_base = workspace_addr;
    state->control.workspace_bytes = kWorkloadBytes;
    state->control.qk_repeats = 1U;
    state->control.sf_repeats = 1U;
    state->control.pv_repeats = 1U;
    state->control.up_repeats = 1U;
    state->control.builder_count = 1;
    state->fatal.state = 0;

    /* init guards (simplified — just zero them) */
    std::memset(&state->guard_before_tasks, 0, sizeof(state->guard_before_tasks));
    std::memset(&state->guard_after_tasks, 0, sizeof(state->guard_after_tasks));

    /* init tasks */
    uint32_t task_count = TaskCount(batches);
    for (uint32_t i = 0; i < kMaxTasks; i++) {
        FullPaTask *task = &state->tasks[i];
        std::memset(&task->plan, 0xD3, sizeof(task->plan));
        task->completion.flag = 0;
        task->completion.vend = 0;
        task->completion.deps_prepared = (int64_t)i - 1;
        task->insert_completion.value = (int64_t)i - 1;
        task->allocation.task_base_plus_one.value = 0;
        task->allocation.completion_vend_plus_one.value = 0;

        for (uint32_t s = 0; s < kOutputsPerTask; s++) {
            task->outputs.published[s].value = (int64_t)-1;
            task->outputs.last_writer[s].value = (int64_t)-1;
            std::memset(&task->outputs.tensors[s], 0, sizeof(task->outputs.tensors[s]));
        }

        task->exec.control.state = 0;
        for (uint32_t w = 0; w < kMaxPayloadWords; w++)
            task->exec.payload.words[w] = 0;

        std::memset(&task->build_report, 0xD3, sizeof(task->build_report));
        std::memset(&task->execution_witness, 0xD3, sizeof(task->execution_witness));

        if (i < task_count) {
            task->plan.task_id = i;
            task->plan.batch = i / kTasksPerBatch;
            task->plan.kind = TaskKindAt(i);
            task->plan.engine_class = TaskEngine(task->plan.kind);
            task->build_report.build_attempt_count = 0;
            task->build_report.build_win_count = 0;
            task->execution_witness.state = 0;

            /* set BuiltState so executor can claim immediately */
            task->exec.control.state = (int64_t)BuiltState(i, kBuilderOwner);
        }
    }

    /* init drain */
    state->drain.builder_started.value = 1;
    state->drain.builder_finished.value = 1;
    state->drain.done_count.value = 0;
    state->drain.alloc_done.value = batches;
    state->drain.aic_done.value = 0;
    state->drain.aiv_done.value = 0;
    state->drain.root_finished.value = 0;
    for (uint32_t g = 0; g < kDrainGroupCount; g++)
        state->drain.arrivals[g].value = 0;

    /* init dispatch */
    state->exec_dispatch.aic_next.value = 0;
    state->exec_dispatch.aiv_next.value = 0;
    state->exec_dispatch.aic_task_count = batches * 2U;
    state->exec_dispatch.aiv_task_count = batches * 2U;
    for (uint32_t i = 0; i < batches * 2U; i++) {
        state->exec_dispatch.aic_task_ids[i] = AicDispatchTaskId(i);
        state->exec_dispatch.aiv_task_ids[i] = AivDispatchTaskId(i);
    }

    /* init heap */
    for (uint32_t s = 0; s < kSharedHeapShards; s++)
        state->heap.shard_cursors[s].value = 0;
    state->heap.aggregate_vend.value = 0;

    /* init tokens */
    for (uint32_t o = 0; o < kOwnerCount; o++) {
        for (uint32_t s = 0; s < kTokensPerOwner; s++) {
            state->tokens[o][s].control.phase = ExecTokenPhase::Idle;
            state->tokens[o][s].control.task_id = UINT32_MAX;
            state->tokens[o][s].control.build_owner = UINT32_MAX;
            state->tokens[o][s].control.execute_owner = UINT32_MAX;
            state->tokens[o][s].control.engine_class = ExecEngineClass::None;
        }
        std::memset(&state->roles[o], 0xD3, sizeof(state->roles[o]));
    }
}

/* CANN KernelArgs layout for AICPU (matches CANN standard) */
struct KernelArgs {
    uint64_t _pad[5];       /* 0..39 */
    void *device_args;      /* 40: pointer to DeviceArgs in GM */
};

struct DeviceArgs {
    uint64_t reserved[12];  /* 0..95 (bootstrap uses 96..128) */
    uint64_t state_addr;    /* 96: ScState address in GM */
    uint64_t workspace_addr;/* 104: workspace address in GM */
};

} /* namespace */

extern "C" {

__attribute__((visibility("default")))
int simpler_aicpu_init(void *args) {
    (void)args;
    return 0;
}

__attribute__((visibility("default")))
int simpler_aicpu_run(void *args) {
    auto *k = reinterpret_cast<KernelArgs *>(args);
    if (!k || !k->device_args) return 1;
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (!d || !d->state_addr) return 1;

    FullPaState *state = reinterpret_cast<FullPaState *>(d->state_addr);
    uint64_t nonce = 0xA5A50001;
    init_state(state, nonce, 1, d->workspace_addr);  /* 1 batch = 5 tasks */

    return 0;
}

} /* extern "C" */
