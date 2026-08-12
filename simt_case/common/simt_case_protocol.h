#ifndef SIMT_CASE_PROTOCOL_H
#define SIMT_CASE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SC_RING_SIZE 4096
#define SC_RING_MASK (SC_RING_SIZE - 1)
#define SC_WARP_SIZE 32
#define SC_NODE_BUFF_SIZE 65536
#define SC_CON_NODE_CNT 30
#define SC_EXE_CORES 93   /* 31 AIC + 62 AIV executor */
#define SC_EXE_OSTD 2     /* ping-pong slots per core */

#ifndef SC_WARP_COUNT
#define SC_WARP_COUNT 8
#endif

/* task type */
#define SC_TT_CUBE   0
#define SC_TT_VECTOR 1

/* task state */
#define SC_TS_CREATING  0
#define SC_TS_COMPLETED 1

/* executor slot state */
#define SC_SLOT_IDLE    0
#define SC_SLOT_CLAIMED 1
#define SC_SLOT_DONE    2

/* warp role assignment (configurable) */
#define SC_DESC_WARPS    (SC_WARP_COUNT / 2)
#define SC_CUTTER_WARP   (SC_DESC_WARPS)
#define SC_DISPATCH_WARP (SC_DESC_WARPS + 1)

/* Qwen3 decode constants */
#define SC_USER_BATCH    90
#define SC_BATCH_PADDED  96
#define SC_NUM_TILES     6
#define SC_Q_PROJ_CHUNKS 20
#define SC_KV_CHUNKS     8
#define SC_ATTN_CHUNKS   4
#define SC_OUT_CHUNKS    40
#define SC_MLP_CHUNKS    34
#define SC_DOWN_CHUNKS   40
#define SC_MAX_TASKS     4096

/* durations (ns / 100) */
#define DUR_RMSNORM       240
#define DUR_Q_PROJ        261
#define DUR_K_PROJ        182
#define DUR_V_PROJ        179
#define DUR_QK_NORM       132
#define DUR_ROPE          95
#define DUR_QK_MATMUL     294
#define DUR_SOFTMAX       194
#define DUR_SV_MATMUL     317
#define DUR_ONLINE_SOFTMAX 208
#define DUR_OUT_PROJ      408
#define DUR_POST_RMSNORM  244
#define DUR_GATE_PROJ     957
#define DUR_UP_PROJ       971
#define DUR_SILU          28
#define DUR_DOWN_PROJ     722
#define DUR_DOWN_PROJ_RES 26

/* ── GM structures ── */

struct alignas(64) ScAtomicLine {
    volatile int64_t value;
    uint8_t pad[56];
};

struct GmTaskDesc {
    uint32_t id;
    uint32_t type;       /* SC_TT_CUBE or SC_TT_VECTOR */
    uint32_t count;      /* SPMD block count */
    uint32_t duration;
    uint32_t tensor_cnt;
    uint32_t scalar_cnt;
    uint32_t pad0[2];
    uint64_t data[8];
    int64_t  scalar[4];
};

struct GmPredList {
    uint32_t cnt;
    uint32_t exp_offset;
};

struct alignas(64) GmSuccList {
    uint32_t cnt;
    uint32_t node[SC_CON_NODE_CNT];
    uint32_t overflow;
    uint32_t pad;
};

struct GmTaskState {
    uint32_t state;
    uint32_t succ_cnt;
};

struct alignas(64) GmQueue {
    volatile uint64_t head;
    volatile uint64_t tail;
    volatile uint64_t cnt;
    uint32_t pad0[5];
    uint32_t tasks[SC_RING_SIZE];
    volatile int64_t lock;
    uint8_t pad1[24];
};

struct alignas(64) GmExeSlot {
    volatile int64_t state;    /* SC_SLOT_IDLE / CLAIMED / DONE */
    uint32_t task_id;
    uint32_t pad;
};

struct ScState {
    /* sync */
    ScAtomicLine phase;
    ScAtomicLine desc_done;
    ScAtomicLine all_done;
    ScAtomicLine completed_cnt;
    ScAtomicLine pred_ring_cursor;

    uint32_t total_task_cnt;
    uint32_t num_cube_tasks;
    uint32_t num_vec_tasks;
    uint32_t pad0[5];

    /* DAG data */
    GmTaskDesc basic_buf[SC_RING_SIZE];
    uint32_t pred_ring[SC_NODE_BUFF_SIZE];
    GmPredList predecessors[SC_RING_SIZE];
    GmSuccList successor_buf[SC_RING_SIZE];
    GmTaskState state_buf[SC_RING_SIZE];
    uint32_t pred_cnt[SC_RING_SIZE];

    /* queues: [0]=CUBE, [1]=VECTOR */
    GmQueue ready_queue[2];
    GmQueue completed_queue;

    /* executor slots: SC_EXE_CORES cores × SC_EXE_OSTD slots */
    GmExeSlot exe_slots[SC_EXE_CORES * SC_EXE_OSTD];

    /* dispatch bitmap: which cores are free */
    uint64_t free_bitmap[2][2];   /* [type][slot] */
    uint64_t msg_bitmap[2][2];    /* [type][slot] completion */

    /* report */
    uint64_t report_desc_writes;
    uint64_t report_cutter_ready;
    uint64_t report_dispatch_sent;
    uint64_t report_executor_done;
    uint64_t report_start_tick;
    uint64_t report_end_tick;
    uint64_t report_magic;
    uint64_t report_pad[7];

    /* workspace for PTO anchor (forces MIX + TLV 0x10/0x11) */
    uint64_t workspace_base;
    uint64_t workspace_bytes;
    uint64_t workspace_pad[6];
};

#ifdef __cplusplus
}
#endif

#endif
