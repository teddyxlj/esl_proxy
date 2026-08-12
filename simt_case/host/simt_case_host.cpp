#define _POSIX_C_SOURCE 200809L

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../common/g0_full_pa.h"

using namespace pa_scheduler::simt_cross_core::g0;

namespace {

constexpr uint64_t kHostWorkloadBytes = static_cast<uint64_t>(kWorkloadTiles) * kWorkloadTileBytes;

bool CheckAcl(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        std::fprintf(stderr, "[ACL ERROR] %s: code=%d\n", msg, (int)err);
        return false;
    }
    return true;
}

bool ReadBinary(const char *path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open: %s\n", path); return false; }
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    out->resize(sz);
    if (sz > 0) f.read(out->data(), (std::streamsize)sz);
    return true;
}

struct Options {
    const char *kernel_path = "build/simt_case_kernel.o";
    int device = 0;
    uint32_t batches = 4;
    uint32_t builders = 1;
    uint32_t runs = 1;
};

bool ParseOptions(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i+1 < argc) opt->device = std::atoi(argv[++i]);
        else if (a == "--kernel" && i+1 < argc) opt->kernel_path = argv[++i];
        else if (a == "--batches" && i+1 < argc) opt->batches = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--builders" && i+1 < argc) opt->builders = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--runs" && i+1 < argc) opt->runs = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: simt_case_host --kernel <path> --device <N> --batches <N> --builders <N> --runs <N>\n");
            return false;
        }
    }
    return true;
}

void InitializeState(FullPaState *state, uint64_t nonce, uint32_t batches, uint32_t builder_count, uint64_t workspace_addr) {
    std::memset(state, 0, sizeof(*state));
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = 5000000000ULL;
    state->control.batch_count = batches;
    state->control.task_count = TaskCount(batches);
    state->control.kernel_task_count = KernelTaskCount(batches);
    state->control.builder_thread_count = kBuilderThreadCount;
    state->control.heap_base = kSyntheticHeapBase;
    state->control.heap_bytes = kHeapBytes;
    state->control.workspace_base = workspace_addr;
    state->control.workspace_bytes = kHostWorkloadBytes;
    state->control.qk_repeats = 1U;
    state->control.sf_repeats = 1U;
    state->control.pv_repeats = 1U;
    state->control.up_repeats = 1U;
    state->control.builder_count = builder_count;
    state->fatal.state = 0;

    for (uint32_t i = 0; i < batches * 2U; ++i) {
        state->exec_dispatch.aic_task_ids[i] = AicDispatchTaskId(i);
        state->exec_dispatch.aiv_task_ids[i] = AivDispatchTaskId(i);
    }
    state->exec_dispatch.aic_task_count = batches * 2U;
    state->exec_dispatch.aiv_task_count = batches * 2U;

    for (uint32_t owner = 0; owner < kOwnerCount; ++owner) {
        for (uint32_t slot = 0; slot < kTokensPerOwner; ++slot) {
            auto *token = &state->tokens[owner][slot];
            token->control.phase = ExecTokenPhase::Idle;
            token->control.task_id = UINT32_MAX;
        }
    }
}

void InitializeWorkspace(std::vector<float> *ws) {
    ws->assign(static_cast<size_t>(kWorkloadTiles) * kWorkloadTileElements, kWorkloadOutputSentinel);
    std::fill_n(ws->data(), kWorkloadTileElements, kWorkloadInputA);
    std::fill_n(ws->data() + kWorkloadTileElements, kWorkloadTileElements, kWorkloadInputB);
}

}

int main(int argc, char **argv) {
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) return 1;

    std::vector<char> bin;
    if (!ReadBinary(opt.kernel_path, &bin)) return 1;

    if (!CheckAcl(aclInit(nullptr), "aclInit")) return 1;
    if (!CheckAcl(aclrtSetDevice(opt.device), "aclrtSetDevice")) return 1;

    const char *soc = aclrtGetSocName();
    std::printf("[DEVICE] id=%d soc=%s batches=%u builders=%u\n", opt.device, soc ? soc : "(null)", opt.batches, opt.builders);
    if (!soc || std::string(soc).rfind("Ascend950", 0) != 0) {
        std::fprintf(stderr, "[ERROR] requires Ascend950\n");
        return 1;
    }

    aclrtStream stream = nullptr;
    aclrtEvent ev0 = nullptr, ev1 = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtCreateEvent(&ev0), "ev0") ||
        !CheckAcl(aclrtCreateEvent(&ev1), "ev1")) return 1;

    aclrtBinHandle bin_handle = nullptr;
    aclrtFuncHandle func_handle = nullptr;
    aclrtBinaryLoadOption load_opt{};
    load_opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    load_opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions load_opts{&load_opt, 1U};
    if (!CheckAcl(aclrtBinaryLoadFromData(bin.data(), bin.size(), &load_opts, &bin_handle), "aclrtBinaryLoadFromData") ||
        !CheckAcl(aclrtBinaryGetFunctionByEntry(bin_handle, 0, &func_handle), "aclrtBinaryGetFunctionByEntry")) return 1;

    void *dev_state = nullptr;
    void *dev_workspace = nullptr;
    if (!CheckAcl(aclrtMalloc(&dev_state, sizeof(FullPaState), ACL_MEM_MALLOC_HUGE_FIRST), "malloc(state)")) return 1;
    if (!CheckAcl(aclrtMalloc(&dev_workspace, kHostWorkloadBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc(workspace)")) return 1;

    std::printf("[ALLOC] state=%p (%zuB) workspace=%p (%lluB)\n",
        dev_state, sizeof(FullPaState), dev_workspace, (unsigned long long)kHostWorkloadBytes);

    uint32_t passes = 0;
    std::vector<double> times_us;

    for (uint32_t run = 0; run < opt.runs; ++run) {
        auto host_state = std::make_unique<FullPaState>();
        uint64_t nonce = UINT64_C(0xA550000000000000) ^ ((uint64_t)opt.batches << 32) ^ ((uint64_t)opt.builders << 24) ^ (uint64_t)(run + 1);
        InitializeState(host_state.get(), nonce, opt.batches, opt.builders, (uint64_t)dev_workspace);

        std::vector<float> host_workspace;
        InitializeWorkspace(&host_workspace);

        if (!CheckAcl(aclrtMemcpy(dev_state, sizeof(*host_state), host_state.get(), sizeof(*host_state), ACL_MEMCPY_HOST_TO_DEVICE), "H2D state")) return 1;
        if (!CheckAcl(aclrtMemcpy(dev_workspace, kHostWorkloadBytes, host_workspace.data(), kHostWorkloadBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D ws")) return 1;

        struct KernelArgs { uint64_t state_ptr; } args{(uint64_t)dev_state};

        if (!CheckAcl(aclrtRecordEvent(ev0, stream), "ev0") ||
            !CheckAcl(aclrtLaunchKernelWithHostArgs(func_handle, 32U, stream, nullptr, &args, sizeof(args), nullptr, 0), "launch") ||
            !CheckAcl(aclrtRecordEvent(ev1, stream), "ev1") ||
            !CheckAcl(aclrtSynchronizeStream(stream), "sync")) return 1;

        float ms = 0;
        CheckAcl(aclrtEventElapsedTime(&ms, ev0, ev1), "elapsed");

        if (!CheckAcl(aclrtMemcpy(host_state.get(), sizeof(*host_state), dev_state, sizeof(*host_state), ACL_MEMCPY_DEVICE_TO_HOST), "D2H state")) return 1;
        if (!CheckAcl(aclrtMemcpy(host_workspace.data(), kHostWorkloadBytes, dev_workspace, kHostWorkloadBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H ws")) return 1;

        bool ok = host_state->drain.root_finished.value == 1U;
        uint32_t total_kernel_tasks = opt.batches * 4U;
        uint32_t done_count = (uint32_t)host_state->drain.done_count.value;
        if (done_count != total_kernel_tasks) {
            std::fprintf(stderr, "[FAIL] run=%u done=%u expected=%u fatal=%lld\n",
                run+1, done_count, total_kernel_tasks, (long long)host_state->fatal.state);
            ok = false;
        }

        if (ok) {
            ++passes;
            times_us.push_back(ms * 1000.0);
            std::printf("[PASS] run=%u batches=%u tasks=%u done=%u kernel_us=%.1f\n",
                run+1, opt.batches, total_kernel_tasks, done_count, ms * 1000.0);
        } else {
            std::printf("[FAIL] run=%u fatal=0x%llx done=%u\n",
                run+1, (unsigned long long)host_state->fatal.state, done_count);
        }
    }

    if (!times_us.empty()) {
        std::sort(times_us.begin(), times_us.end());
        std::printf("[PERF] passes=%u/%u median_us=%.1f min_us=%.1f max_us=%.1f\n",
            passes, opt.runs, times_us[times_us.size()/2], times_us.front(), times_us.back());
    }

    aclrtFree(dev_workspace);
    aclrtFree(dev_state);
    aclrtDestroyEvent(ev0);
    aclrtDestroyEvent(ev1);
    aclrtDestroyStream(stream);
    aclrtBinaryUnLoad(bin_handle);
    aclrtResetDevice(opt.device);
    return passes == opt.runs ? 0 : 1;
}
