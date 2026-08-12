#define _POSIX_C_SOURCE 200809L

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/simt_case_protocol.h"

/* ── Qwen3 task graph builder (mirrors desc_ap.h tier 0) ── */

namespace {

struct HostTask {
    uint32_t type;
    uint32_t duration;
};

struct HostGraph {
    HostTask tasks[SC_MAX_TASKS];
    uint32_t pred_ring[SC_NODE_BUFF_SIZE];
    struct { uint32_t cnt; uint32_t offset; } preds[SC_MAX_TASKS];
    uint32_t task_count = 0;
    uint32_t pred_count = 0;

    uint32_t add(uint32_t type, uint32_t dur) {
        uint32_t id = task_count++;
        tasks[id].type = type;
        tasks[id].duration = dur;
        preds[id].cnt = 0;
        preds[id].offset = pred_count;
        return id;
    }
    void pred(uint32_t tid, uint32_t pid) {
        if (preds[tid].cnt == 0) preds[tid].offset = pred_count;
        pred_ring[pred_count++] = pid;
        preds[tid].cnt++;
    }
};

void build_qwen3_graph(HostGraph &g) {
    g.task_count = 0; g.pred_count = 0;
    uint32_t rms_ids[6], qk_norm_ids[6];
    uint32_t v_ids[6][8]; uint32_t v_cnt[6];
    uint32_t os_ids[90][4]; uint32_t os_cnt[90];
    uint32_t op_ids[6][40]; uint32_t op_cnt[6];
    uint32_t silu_ids[6][34]; uint32_t mlp_cnt[6];
    for (uint32_t t=0;t<6;t++){v_cnt[t]=0;op_cnt[t]=0;mlp_cnt[t]=0;}
    for (uint32_t r=0;r<90;r++) os_cnt[r]=0;

    for (uint32_t b0=0;b0<96;b0+=16) {
        uint32_t t=b0/16;
        rms_ids[t]=g.add(SC_TT_VECTOR,DUR_RMSNORM);
        for(uint32_t b=0;b<20;b++){uint32_t id=g.add(SC_TT_CUBE,DUR_Q_PROJ);g.pred(id,rms_ids[t]);}
        for(uint32_t b=0;b<8;b++){
            uint32_t k=g.add(SC_TT_CUBE,DUR_K_PROJ);g.pred(k,rms_ids[t]);
            uint32_t v=g.add(SC_TT_CUBE,DUR_V_PROJ);g.pred(v,rms_ids[t]);
            v_ids[t][v_cnt[t]++]=v;
        }
        qk_norm_ids[t]=g.add(SC_TT_VECTOR,DUR_QK_NORM);
        for(uint32_t i=0;i<20;i++) g.pred(qk_norm_ids[t],rms_ids[t]+1+i);
        for(uint32_t i=0;i<8;i++) g.pred(qk_norm_ids[t],rms_ids[t]+1+20+i*2);
    }
    for(uint32_t b=0;b<90;b++){
        uint32_t t=b/16;
        uint32_t rope=g.add(SC_TT_VECTOR,DUR_ROPE);
        g.pred(rope,qk_norm_ids[t]);
        for(uint32_t i=0;i<v_cnt[t];i++) g.pred(rope,v_ids[t][i]);
        for(uint32_t base=0;base<4;base++){
            uint32_t qk=g.add(SC_TT_CUBE,DUR_QK_MATMUL);g.pred(qk,rope);
            uint32_t sm=g.add(SC_TT_VECTOR,DUR_SOFTMAX);g.pred(sm,qk);
            uint32_t sv=g.add(SC_TT_CUBE,DUR_SV_MATMUL);g.pred(sv,rope);g.pred(sv,sm);
            uint32_t os=g.add(SC_TT_VECTOR,DUR_ONLINE_SOFTMAX);g.pred(os,sv);g.pred(os,sm);
            os_ids[b][os_cnt[b]++]=os;
        }
    }
    for(uint32_t b0=0;b0<96;b0+=16){
        uint32_t t=b0/16;
        uint32_t cv=(90-b0>16)?16:(90-b0);
        for(uint32_t b=0;b<40;b++){
            uint32_t id=g.add(SC_TT_CUBE,DUR_OUT_PROJ);op_ids[t][op_cnt[t]++]=id;
            for(uint32_t row=0;row<cv;row++){uint32_t bb=b0+row;for(uint32_t i=0;i<os_cnt[bb];i++)g.pred(id,os_ids[bb][i]);}
        }
        uint32_t post=g.add(SC_TT_VECTOR,DUR_POST_RMSNORM);
        for(uint32_t i=0;i<op_cnt[t];i++) g.pred(post,op_ids[t][i]);
        for(uint32_t b=0;b<34;b++){
            uint32_t gid=g.add(SC_TT_CUBE,DUR_GATE_PROJ);g.pred(gid,post);
            uint32_t uid=g.add(SC_TT_CUBE,DUR_UP_PROJ);g.pred(uid,post);
            uint32_t sid=g.add(SC_TT_VECTOR,DUR_SILU);g.pred(sid,gid);g.pred(sid,uid);
            silu_ids[t][mlp_cnt[t]++]=sid;
        }
        for(uint32_t b=0;b<40;b++){
            uint32_t did=g.add(SC_TT_CUBE,DUR_DOWN_PROJ);
            for(uint32_t i=0;i<mlp_cnt[t];i++) g.pred(did,silu_ids[t][i]);
            uint32_t dr=g.add(SC_TT_VECTOR,DUR_DOWN_PROJ_RES);
            g.pred(dr,did);g.pred(dr,op_ids[t][b<op_cnt[t]?b:op_cnt[t]-1]);
        }
    }
}

/* ── ACL helpers ── */

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

/* ── Host-side cutter + dispatch + executor ── */

void host_cutter_dispatch_execute(ScState *st, const HostGraph &g) {
    uint32_t total = g.task_count;

    /* cutter: build successor lists */
    for (uint32_t i = 0; i < total; i++) {
        if (g.preds[i].cnt == 0) {
            uint32_t type = g.tasks[i].type;
            if (type >= 2) type = SC_TT_VECTOR;
            /* enqueue to ready_queue */
            GmQueue *q = &st->ready_queue[type];
            q->tasks[q->tail % SC_RING_SIZE] = i;
            q->tail++; q->cnt++;
            st->pred_cnt[i] = 0;
        } else {
            uint32_t pending = 0;
            for (uint32_t j = 0; j < g.preds[i].cnt; j++) {
                uint32_t pred = g.pred_ring[g.preds[i].offset + j];
                if (st->state_buf[pred].state != SC_TS_COMPLETED) {
                    uint32_t sidx = st->successor_buf[pred].cnt;
                    if (sidx < SC_CON_NODE_CNT)
                        st->successor_buf[pred].node[sidx] = i;
                    st->successor_buf[pred].cnt++;
                    st->state_buf[pred].succ_cnt++;
                    pending++;
                }
            }
            st->pred_cnt[i] = pending;
            if (pending == 0) {
                uint32_t type = g.tasks[i].type;
                if (type >= 2) type = SC_TT_VECTOR;
                GmQueue *q = &st->ready_queue[type];
                q->tasks[q->tail % SC_RING_SIZE] = i;
                q->tail++; q->cnt++;
            }
        }
    }
    st->report_cutter_ready = total;

    /* dispatch + execute: immediate complete (no real AICore execution) */
    uint32_t done = 0;
    while (done < total) {
        for (uint32_t t = 0; t < 2; t++) {
            GmQueue *q = &st->ready_queue[t];
            uint32_t tid;
            while (q->cnt > 0) {
                tid = q->tasks[q->head % SC_RING_SIZE];
                q->head++; q->cnt--;

                /* immediate complete */
                st->state_buf[tid].state = SC_TS_COMPLETED;
                done++;

                /* resolve deps */
                uint32_t sc = st->state_buf[tid].succ_cnt;
                for (uint32_t k = 0; k < sc; k++) {
                    uint32_t succ = st->successor_buf[tid].node[k];
                    if (--st->pred_cnt[succ] == 0) {
                        uint32_t stype = g.tasks[succ].type;
                        if (stype >= 2) stype = SC_TT_VECTOR;
                        GmQueue *rq = &st->ready_queue[stype];
                        rq->tasks[rq->tail % SC_RING_SIZE] = succ;
                        rq->tail++; rq->cnt++;
                    }
                }
            }
        }
    }
    st->report_dispatch_sent = total;
    st->report_executor_done = total;
    st->completed_cnt.value = total;
}

} /* namespace */

int main(int argc, char **argv) {
    const char *kernel_path = "build/simt_case_kernel.o";
    int device = 0;
    uint32_t runs = 1;
    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if(a=="--device"&&i+1<argc) device=std::atoi(argv[++i]);
        else if(a=="--kernel"&&i+1<argc) kernel_path=argv[++i];
        else if(a=="--runs"&&i+1<argc) runs=(uint32_t)std::atoi(argv[++i]);
        else if(a=="--help"){std::printf("Usage: --kernel <path> --device <N> --runs <N>\n");return 1;}
    }

    /* build Qwen3 graph */
    HostGraph graph;
    build_qwen3_graph(graph);
    uint32_t cube_n=0, vec_n=0;
    for(uint32_t i=0;i<graph.task_count;i++){if(graph.tasks[i].type==SC_TT_CUBE)cube_n++;else vec_n++;}
    std::printf("[INIT] Qwen3-14B decode: %u tasks, %u edges, %u cube + %u vec\n",
        graph.task_count, graph.pred_count, cube_n, vec_n);

    /* read kernel binary */
    std::vector<char> bin;
    if (!ReadBinary(kernel_path, &bin)) return 1;

    /* ACL init */
    if (!CheckAcl(aclInit(nullptr), "aclInit")) return 1;
    if (!CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) return 1;
    const char *soc = aclrtGetSocName();
    std::printf("[DEVICE] id=%d soc=%s\n", device, soc ? soc : "(null)");

    aclrtStream stream=nullptr; aclrtEvent ev0=nullptr, ev1=nullptr;
    if(!CheckAcl(aclrtCreateStream(&stream),"stream")||
       !CheckAcl(aclrtCreateEvent(&ev0),"ev0")||
       !CheckAcl(aclrtCreateEvent(&ev1),"ev1")) return 1;

    /* load binary */
    aclrtBinHandle bin_h=nullptr; aclrtFuncHandle func_h=nullptr;
    aclrtBinary binary = aclrtCreateBinary(bin.data(), bin.size());
    if (!binary) { std::fprintf(stderr,"[ERROR] aclrtCreateBinary null\n"); return 1; }
    if(!CheckAcl(aclrtBinaryLoad(binary,&bin_h),"load")||
       !CheckAcl(aclrtBinaryGetFunctionByEntry(bin_h,0,&func_h),"getfunc")) return 1;

    /* alloc GM */
    void *dev_state=nullptr;
    void *dev_ws=nullptr;
    constexpr uint64_t kWsBytes = 3 * 128 * 128 * sizeof(float); /* 192KB — 3 tiles */
    if(!CheckAcl(aclrtMalloc(&dev_state,sizeof(ScState),ACL_MEM_MALLOC_HUGE_FIRST),"malloc state")) return 1;
    if(!CheckAcl(aclrtMalloc(&dev_ws,kWsBytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc ws")) return 1;
    std::printf("[ALLOC] state=%p (%zuB) ws=%p (%lluB)\n", dev_state, sizeof(ScState), dev_ws, (unsigned long long)kWsBytes);

    uint32_t passes=0;
    std::vector<double> times_us;

    for (uint32_t run=0; run<runs; ++run) {
        /* build host-side ScState */
        auto hs = std::make_unique<ScState>();
        std::memset(hs.get(), 0, sizeof(ScState));
        hs->total_task_cnt = graph.task_count;
        hs->workspace_base = (uint64_t)dev_ws;
        hs->workspace_bytes = kWsBytes;

        for (uint32_t i=0; i<graph.task_count; i++) {
            hs->basic_buf[i].id = i;
            hs->basic_buf[i].type = graph.tasks[i].type;
            hs->basic_buf[i].count = 1;
            hs->basic_buf[i].duration = graph.tasks[i].duration;
            hs->predecessors[i].cnt = graph.preds[i].cnt;
            hs->predecessors[i].exp_offset = graph.preds[i].offset;
            for (uint32_t j=0; j<graph.preds[i].cnt; j++)
                hs->pred_ring[graph.preds[i].offset+j] = graph.pred_ring[graph.preds[i].offset+j];
            hs->state_buf[i].state = SC_TS_CREATING;
        }

        /* H2D: send state to device (desc_done=0, all_done=0) */
        if(!CheckAcl(aclrtMemcpy(dev_state,sizeof(ScState),hs.get(),sizeof(ScState),ACL_MEMCPY_HOST_TO_DEVICE),"H2D")) return 1;

        /* launch kernel: SIMT desc VF on AIV0, AIC/AIV wait for all_done */
        struct KArgs { uint64_t ptr; } args{(uint64_t)dev_state};
        if(!CheckAcl(aclrtRecordEvent(ev0,stream),"ev0")||
           !CheckAcl(aclrtLaunchKernelWithHostArgs(func_h,32U,stream,nullptr,&args,sizeof(args),nullptr,0),"launch")||
           !CheckAcl(aclrtRecordEvent(ev1,stream),"ev1")) return 1;

        /* wait for SIMT desc to finish (poll desc_done on device) */
        while (true) {
            ScState peek;
            if(!CheckAcl(aclrtMemcpy(&peek,sizeof(ScState),dev_state,sizeof(ScState),ACL_MEMCPY_DEVICE_TO_HOST),"peek")) return 1;
            if (peek.desc_done.value == 1) break;
            std::this_thread::yield();
        }

        /* host-side cutter + dispatch + executor (immediate complete) */
        /* need to D2H the state (desc VF wrote state_buf), run cutter, then H2D back */
        if(!CheckAcl(aclrtMemcpy(hs.get(),sizeof(ScState),dev_state,sizeof(ScState),ACL_MEMCPY_DEVICE_TO_HOST),"D2H-desc")) return 1;

        host_cutter_dispatch_execute(hs.get(), graph);

        /* H2D: write all_done=1 to stop kernel */
        if(!CheckAcl(aclrtMemcpy(dev_state,sizeof(ScState),hs.get(),sizeof(ScState),ACL_MEMCPY_HOST_TO_DEVICE),"H2D-done")) return 1;

        if(!CheckAcl(aclrtSynchronizeStream(stream),"sync")) return 1;

        float ms=0;
        CheckAcl(aclrtEventElapsedTime(&ms,ev0,ev1),"elapsed");

        /* D2H final */
        if(!CheckAcl(aclrtMemcpy(hs.get(),sizeof(ScState),dev_state,sizeof(ScState),ACL_MEMCPY_DEVICE_TO_HOST),"D2H-final")) return 1;

        bool ok = hs->all_done.value == 1 && hs->desc_done.value == 1;
        uint32_t desc_w = (uint32_t)hs->report_desc_writes;
        if (desc_w != graph.task_count) {
            std::fprintf(stderr,"[FAIL] run=%u desc_writes=%u expected=%u\n",run+1,desc_w,graph.task_count);
            ok = false;
        }

        if (ok) {
            ++passes;
            times_us.push_back(ms*1000.0);
            std::printf("[PASS] run=%u tasks=%u desc=%u kernel_us=%.1f\n",
                run+1, graph.task_count, desc_w, ms*1000.0);
        } else {
            std::printf("[FAIL] run=%u all_done=%lld desc=%llu\n",
                run+1,(long long)hs->all_done.value,(unsigned long long)hs->report_desc_writes);
        }
    }

    if(!times_us.empty()){
        std::sort(times_us.begin(),times_us.end());
        std::printf("[PERF] passes=%u/%u median_us=%.1f min_us=%.1f max_us=%.1f\n",
            passes,runs,times_us[times_us.size()/2],times_us.front(),times_us.back());
    }

    aclrtFree(dev_ws);
    aclrtFree(dev_state);
    aclrtDestroyEvent(ev0); aclrtDestroyEvent(ev1);
    aclrtDestroyStream(stream);
    aclrtBinaryUnLoad(bin_h);
    aclrtResetDevice(device);
    return passes==runs?0:1;
}
