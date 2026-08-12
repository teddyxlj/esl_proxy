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
#define SC_EXE_CORES 93
#define SC_EXE_OSTD 2
#define SC_MAX_TASKS 4096

#ifndef SC_WARP_COUNT
#define SC_WARP_COUNT 8
#endif

#define SC_TT_CUBE   0
#define SC_TT_VECTOR 1

#define SC_TS_CREATING  0
#define SC_TS_COMPLETED 1

#define SC_SLOT_IDLE    0
#define SC_SLOT_CLAIMED 1
#define SC_SLOT_DONE    2

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

struct alignas(64) ScAtomicLine {
    volatile int64_t value;
    uint8_t pad[56];
};

struct GmTaskDesc {
    uint32_t id;
    uint32_t type;
    uint32_t count;
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
    volatile int64_t state;
    uint32_t task_id;
    uint32_t pad;
};

/* AICPU → AICore args (passed via GM) */
struct ScState {
    ScAtomicLine phase;          /* 0=init, 1=alloc_done, 2=desc_done, 3=all_done */
    ScAtomicLine completed_cnt;
    ScAtomicLine pred_ring_cursor;

    uint32_t total_task_cnt;
    uint32_t num_cube_tasks;
    uint32_t num_vec_tasks;
    uint32_t pad0[5];

    GmTaskDesc basic_buf[SC_RING_SIZE];
    uint32_t pred_ring[SC_NODE_BUFF_SIZE];
    GmPredList predecessors[SC_RING_SIZE];
    GmSuccList successor_buf[SC_RING_SIZE];
    GmTaskState state_buf[SC_RING_SIZE];
    uint32_t pred_cnt[SC_RING_SIZE];

    GmQueue ready_queue[2];      /* [0]=CUBE, [1]=VECTOR */
    GmQueue completed_queue;

    GmExeSlot exe_slots[SC_EXE_CORES * SC_EXE_OSTD];

    uint64_t free_bitmap[2][2];
    uint64_t msg_bitmap[2][2];

    uint64_t report_alloc_done;
    uint64_t report_desc_writes;
    uint64_t report_cutter_ready;
    uint64_t report_dispatch_sent;
    uint64_t report_executor_done;
    uint64_t report_start_tick;
    uint64_t report_end_tick;
};

/* KernelArgs layout for AICPU (CANN standard) */
struct ScKernelArgs {
    uint64_t _pad[5];
    uint64_t state_ptr;     /* offset 40: pointer to ScState in GM */
};

#ifdef __cplusplus
}
#endif

#endif
