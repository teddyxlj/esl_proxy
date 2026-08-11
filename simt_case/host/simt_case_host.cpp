#define _POSIX_C_SOURCE 200809L

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../common/simt_case_protocol.h"

namespace {

bool CheckAcl(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        std::fprintf(stderr, "[ACL ERROR] %s: code=%d\n", msg, (int)err);
        return false;
    }
    return true;
}

bool ReadBinary(const char *path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open kernel binary: %s\n", path);
        return false;
    }
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
    uint32_t total_tasks = 64;
    uint32_t runs = 1;
};

bool ParseOptions(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) {
            opt->device = std::atoi(argv[++i]);
        } else if (a == "--kernel" && i + 1 < argc) {
            opt->kernel_path = argv[++i];
        } else if (a == "--tasks" && i + 1 < argc) {
            opt->total_tasks = (uint32_t)std::atoi(argv[++i]);
        } else if (a == "--runs" && i + 1 < argc) {
            opt->runs = (uint32_t)std::atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: simt_case_host --kernel <path> --device <N> --tasks <N> --runs <N>\n");
            std::printf("  Defaults: kernel=build/simt_case_kernel.o device=0 tasks=64 runs=1\n");
            return false;
        }
    }
    return true;
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
    std::printf("[DEVICE] id=%d soc=%s\n", opt.device, soc ? soc : "(null)");
    if (!soc || std::string(soc).rfind("Ascend950", 0) != 0) {
        std::fprintf(stderr, "[ERROR] requires Ascend950, got: %s\n", soc ? soc : "(null)");
        return 1;
    }

    aclrtStream stream = nullptr;
    aclrtEvent ev_start = nullptr, ev_end = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtCreateEvent(&ev_start), "aclrtCreateEvent(start)") ||
        !CheckAcl(aclrtCreateEvent(&ev_end), "aclrtCreateEvent(end)")) return 1;

    aclrtBinaryLoadOption load_opt;
    std::memset(&load_opt, 0, sizeof(load_opt));
    load_opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    load_opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions load_opts;
    load_opts.options = &load_opt;
    load_opts.numOpt = 1;
    aclrtBinHandle bin_handle = nullptr;
    aclrtFuncHandle func_handle = nullptr;
    aclError load_err = aclrtBinaryLoadFromData(bin.data(), bin.size(), &load_opts, &bin_handle);
    if (load_err != ACL_SUCCESS) {
        std::fprintf(stderr, "[ACL ERROR] aclrtBinaryLoadFromData: code=%d\n", (int)load_err);
        aclrtBinary binary = aclrtCreateBinary(bin.data(), bin.size());
        if (binary != nullptr) {
            load_err = aclrtBinaryLoad(binary, &bin_handle);
            std::fprintf(stderr, "[ACL ERROR] aclrtBinaryLoad fallback: code=%d\n", (int)load_err);
        }
        if (load_err != ACL_SUCCESS) return 1;
    }
    if (!CheckAcl(aclrtBinaryGetFunctionByEntry(bin_handle, 0, &func_handle), "aclrtBinaryGetFunctionByEntry")) return 1;

    void *dev_state = nullptr;
    if (!CheckAcl(aclrtMalloc(&dev_state, sizeof(SimtCaseState), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(state)")) return 1;
    if (((uintptr_t)dev_state & 63) != 0) {
        std::fprintf(stderr, "[ERROR] dev_state not 64B aligned: %p\n", dev_state);
        return 1;
    }

    uint32_t passes = 0;
    std::vector<double> times_us;

    for (uint32_t run = 0; run < opt.runs; run++) {
        SimtCaseState host_state;
        std::memset(&host_state, 0, sizeof(host_state));
        host_state.phase.value = PHASE_DESC;
        host_state.total_task_cnt = opt.total_tasks;

        if (!CheckAcl(aclrtMemcpy(dev_state, sizeof(host_state), &host_state, sizeof(host_state), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)")) return 1;

        struct KernelArgs { uint64_t state_ptr; } args{(uint64_t)dev_state};

        if (!CheckAcl(aclrtRecordEvent(ev_start, stream), "aclrtRecordEvent(start)") ||
            !CheckAcl(aclrtLaunchKernelWithHostArgs(func_handle, 32U, stream, nullptr, &args, sizeof(args), nullptr, 0), "aclrtLaunchKernelWithHostArgs") ||
            !CheckAcl(aclrtRecordEvent(ev_end, stream), "aclrtRecordEvent(end)") ||
            !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) return 1;

        float ms = 0;
        CheckAcl(aclrtEventElapsedTime(&ms, ev_start, ev_end), "aclrtEventElapsedTime");

        if (!CheckAcl(aclrtMemcpy(&host_state, sizeof(host_state), dev_state, sizeof(host_state), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)")) return 1;

        bool ok = true;
        if (host_state.all_done.value != 1) {
            std::fprintf(stderr, "[FAIL] run=%u all_done=%lld\n", run+1, (long long)host_state.all_done.value);
            ok = false;
        }
        if (host_state.report_desc_writes != opt.total_tasks) {
            std::fprintf(stderr, "[FAIL] run=%u desc_writes=%llu expected=%u\n", run+1, (unsigned long long)host_state.report_desc_writes, opt.total_tasks);
            ok = false;
        }
        if (host_state.report_executor_done != opt.total_tasks) {
            std::fprintf(stderr, "[FAIL] run=%u executor_done=%llu expected=%u\n", run+1, (unsigned long long)host_state.report_executor_done, opt.total_tasks);
            ok = false;
        }

        if (ok) {
            passes++;
            times_us.push_back(ms * 1000.0);
            std::printf("[PASS] run=%u tasks=%u desc=%llu cutter=%llu dispatch=%llu exec=%llu kernel_us=%.1f\n",
                run+1, opt.total_tasks,
                (unsigned long long)host_state.report_desc_writes,
                (unsigned long long)host_state.report_cutter_ready,
                (unsigned long long)host_state.report_dispatch_sent,
                (unsigned long long)host_state.report_executor_done,
                ms * 1000.0);
        }
    }

    if (!times_us.empty()) {
        std::sort(times_us.begin(), times_us.end());
        double median = times_us[times_us.size()/2];
        std::printf("[PERF] runs=%u passes=%u/%u median_us=%.1f min_us=%.1f max_us=%.1f\n",
            opt.runs, passes, opt.runs, median, times_us.front(), times_us.back());
    }

    aclrtFree(dev_state);
    aclrtDestroyEvent(ev_start);
    aclrtDestroyEvent(ev_end);
    aclrtDestroyStream(stream);
    aclrtBinaryUnLoad(bin_handle);
    aclrtResetDevice(opt.device);
    return passes == opt.runs ? 0 : 1;
}
