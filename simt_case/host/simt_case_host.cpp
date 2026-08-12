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
#include "../common/qwen3_task_model.h"

using namespace pa_scheduler::simt_cross_core::g0;
using namespace simt_case_qwen3;

namespace {

constexpr uint64_t kHostWorkloadBytes = static_cast<uint64_t>(kWorkloadTiles) * kWorkloadTileBytes;

bool CheckAcl(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) { std::fprintf(stderr,"[ACL ERROR] %s: code=%d\n",msg,(int)err); return false; }
    return true;
}

bool ReadBinary(const char *path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr,"cannot open: %s\n",path); return false; }
    f.seekg(0,std::ios::end); size_t sz=(size_t)f.tellg(); f.seekg(0,std::ios::beg);
    out->resize(sz); if(sz>0) f.read(out->data(),(std::streamsize)sz); return true;
}

struct Options {
    const char *kernel_path = "build/simt_case_kernel.o";
    int device = 0;
    uint32_t runs = 1;
};

bool ParseOptions(int argc, char **argv, Options *opt) {
    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if(a=="--device"&&i+1<argc) opt->device=std::atoi(argv[++i]);
        else if(a=="--kernel"&&i+1<argc) opt->kernel_path=argv[++i];
        else if(a=="--runs"&&i+1<argc) opt->runs=(uint32_t)std::atoi(argv[++i]);
        else if(a=="--help"){std::printf("Usage: --kernel <path> --device <N> --runs <N>\n");return false;}
    }
    return true;
}

void InitializeWorkspace(std::vector<float> *ws) {
    ws->assign(static_cast<size_t>(kWorkloadTiles) * kWorkloadTileElements, kWorkloadOutputSentinel);
    std::fill_n(ws->data(), kWorkloadTileElements, kWorkloadInputA);
    std::fill_n(ws->data() + kWorkloadTileElements, kWorkloadTileElements, kWorkloadInputB);
}

void InitializeStateFromQwen3(FullPaState *state, uint64_t nonce, uint64_t workspace_addr) {
    std::memset(state, 0, sizeof(*state));

    /* use reference's PA task graph: 1 batch = 5 tasks (Alloc, Qk, Sf, Pv, Up) */
    uint32_t batches = 1;
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
    state->control.workspace_bytes = kHostWorkloadBytes;
    state->control.qk_repeats = 1U;
    state->control.sf_repeats = 1U;
    state->control.pv_repeats = 1U;
    state->control.up_repeats = 1U;
    state->control.builder_count = 1;
    state->fatal.state = 0;

    /* init all tasks with reference's PA task graph */
    for (uint32_t i = 0; i < state->control.task_count; i++) {
        FullPaTask *task = &state->tasks[i];
        task->plan.task_id = i;
        task->plan.batch = i / kTasksPerBatch;
        task->plan.kind = TaskKindAt(i);
        task->plan.engine_class = TaskEngine(task->plan.kind);
        task->exec.control.state = 0; /* builder will set BuiltState */
        task->completion.flag = 0;
        task->completion.vend = 0;
        task->execution_witness.state = 0;
    }

    /* dispatch task IDs (reference PA pattern) */
    for (uint32_t i = 0; i < batches * 2U; i++) {
        state->exec_dispatch.aic_task_ids[i] = AicDispatchTaskId(i);
        state->exec_dispatch.aiv_task_ids[i] = AivDispatchTaskId(i);
    }
    state->exec_dispatch.aic_task_count = batches * 2U;
    state->exec_dispatch.aiv_task_count = batches * 2U;

    /* init tokens */
    for (uint32_t owner = 0; owner < kOwnerCount; owner++) {
        for (uint32_t slot = 0; slot < kTokensPerOwner; slot++) {
            state->tokens[owner][slot].control.phase = ExecTokenPhase::Idle;
            state->tokens[owner][slot].control.task_id = UINT32_MAX;
        }
    }
}

} /* namespace */

int main(int argc, char **argv) {
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) return 1;

    /* read kernel binary */
    std::vector<char> bin;
    if (!ReadBinary(opt.kernel_path, &bin)) return 1;

    if (!CheckAcl(aclInit(nullptr), "aclInit")) return 1;
    if (!CheckAcl(aclrtSetDevice(opt.device), "aclrtSetDevice")) return 1;
    const char *soc = aclrtGetSocName();
    std::printf("[DEVICE] id=%d soc=%s\n", opt.device, soc ? soc : "(null)");
    if (!soc || std::string(soc).rfind("Ascend950", 0) != 0) {
        std::fprintf(stderr, "[ERROR] requires Ascend950\n"); return 1;
    }

    aclrtStream stream=nullptr; aclrtEvent ev0=nullptr, ev1=nullptr;
    if(!CheckAcl(aclrtCreateStream(&stream),"stream")||
       !CheckAcl(aclrtCreateEvent(&ev0),"ev0")||
       !CheckAcl(aclrtCreateEvent(&ev1),"ev1")) return 1;

    /* load binary */
    aclrtBinHandle bin_h=nullptr; aclrtFuncHandle func_h=nullptr;
    aclrtBinaryLoadOption lo{}; lo.type=ACL_RT_BINARY_LOAD_OPT_MAGIC;
    lo.value.magic=ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions los{&lo,1U};
    if(!CheckAcl(aclrtBinaryLoadFromData(bin.data(),bin.size(),&los,&bin_h),"load")||
       !CheckAcl(aclrtBinaryGetFunctionByEntry(bin_h,0,&func_h),"getfunc")) return 1;

    /* alloc GM */
    void *dev_state=nullptr; void *dev_ws=nullptr;
    if(!CheckAcl(aclrtMalloc(&dev_state,sizeof(FullPaState),ACL_MEM_MALLOC_HUGE_FIRST),"malloc state")) return 1;
    if(!CheckAcl(aclrtMalloc(&dev_ws,kHostWorkloadBytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc ws")) return 1;
    std::printf("[ALLOC] state=%p (%zuB) ws=%p (%lluB)\n",
        dev_state, sizeof(FullPaState), dev_ws, (unsigned long long)kHostWorkloadBytes);

    /* build Qwen3 graph stats (for reporting only — actual tasks use PA graph) */
    Qwen3TaskGraph graph;
    graph.build_tier0();
    std::printf("[INIT] Qwen3-14B decode graph: %u tasks, %u edges (reference PA binary runs 1 batch=5 tasks)\n",
        graph.task_count, graph.pred_edge_count);

    uint32_t passes=0;
    std::vector<double> times_us;

    for (uint32_t run=0; run<opt.runs; ++run) {
        auto hs = std::make_unique<FullPaState>();
        uint64_t nonce = UINT64_C(0xA5A50000) ^ (uint64_t)(run+1);
        InitializeStateFromQwen3(hs.get(), nonce, (uint64_t)dev_ws);

        std::vector<float> host_ws;
        InitializeWorkspace(&host_ws);

        if(!CheckAcl(aclrtMemcpy(dev_state,sizeof(FullPaState),hs.get(),sizeof(FullPaState),ACL_MEMCPY_HOST_TO_DEVICE),"H2D state")) return 1;
        if(!CheckAcl(aclrtMemcpy(dev_ws,kHostWorkloadBytes,host_ws.data(),kHostWorkloadBytes,ACL_MEMCPY_HOST_TO_DEVICE),"H2D ws")) return 1;

        struct KArgs { uint64_t ptr; } args{(uint64_t)dev_state};
        if(!CheckAcl(aclrtRecordEvent(ev0,stream),"ev0")||
           !CheckAcl(aclrtLaunchKernelWithHostArgs(func_h,32U,stream,nullptr,&args,sizeof(args),nullptr,0),"launch")||
           !CheckAcl(aclrtRecordEvent(ev1,stream),"ev1")||
           !CheckAcl(aclrtSynchronizeStream(stream),"sync")) return 1;

        float ms=0;
        CheckAcl(aclrtEventElapsedTime(&ms,ev0,ev1),"elapsed");

        if(!CheckAcl(aclrtMemcpy(hs.get(),sizeof(FullPaState),dev_state,sizeof(FullPaState),ACL_MEMCPY_DEVICE_TO_HOST),"D2H")) return 1;
        if(!CheckAcl(aclrtMemcpy(host_ws.data(),kHostWorkloadBytes,dev_ws,kHostWorkloadBytes,ACL_MEMCPY_DEVICE_TO_HOST),"D2H ws")) return 1;

        bool ok = hs->drain.root_finished.value == 1U;
        uint32_t done = (uint32_t)hs->drain.done_count.value;
        uint32_t expected = KernelTaskCount(1); /* 1 batch × 4 kernel tasks = 4 */
        if (done != expected) {
            std::fprintf(stderr,"[FAIL] run=%u done=%u expected=%u fatal=0x%llx\n",
                run+1, done, expected, (unsigned long long)hs->fatal.state);
            ok = false;
        }

        if (ok) {
            ++passes;
            times_us.push_back(ms*1000.0);
            std::printf("[PASS] run=%u tasks=%u done=%u kernel_us=%.1f\n",
                run+1, expected, done, ms*1000.0);
        } else {
            std::printf("[FAIL] run=%u fatal=0x%llx done=%u\n",
                run+1, (unsigned long long)hs->fatal.state, done);
        }
    }

    if(!times_us.empty()){
        std::sort(times_us.begin(),times_us.end());
        std::printf("[PERF] passes=%u/%u median_us=%.1f min_us=%.1f max_us=%.1f\n",
            passes,opt.runs,times_us[times_us.size()/2],times_us.front(),times_us.back());
    }

    aclrtFree(dev_ws); aclrtFree(dev_state);
    aclrtDestroyEvent(ev0); aclrtDestroyEvent(ev1);
    aclrtDestroyStream(stream);
    aclrtBinaryUnLoad(bin_h);
    aclrtResetDevice(opt.device);
    return passes==opt.runs?0:1;
}
