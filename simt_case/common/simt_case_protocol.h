#ifndef SIMT_CASE_PROTOCOL_H
#define SIMT_CASE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIMT_CASE_RING_SIZE 4096
#define SIMT_CASE_RING_MASK (SIMT_CASE_RING_SIZE - 1)
#define SIMT_CASE_KWARP_SIZE 32
#define SIMT_CASE_MAX_WARPS 8
#define SIMT_CASE_AIC_OSTD 2
#define SIMT_CASE_AIC_CNT 60
#define SIMT_CASE_EXE_TYPE_CNT 2
#define SIMT_CASE_TASK_TYPE_CNT 3
#define SIMT_CASE_CQ_BATCH_SIZE 512
#define SIMT_CASE_PRE_BATCH_SIZE 240
#define SIMT_CASE_RQ_BATCH_SIZE 512
#define SIMT_CASE_NODE_BUFF_SIZE 65536
#define SIMT_CASE_CON_NODE_CNT 30

#ifndef SIMT_CASE_WARP_COUNT
#define SIMT_CASE_WARP_COUNT 8
#endif

#define PHASE_INIT   0
#define PHASE_ALLOC  1
#define PHASE_DESC   2
#define PHASE_SCHED  3
#define PHASE_DONE   4

#define TASK_TYPE_CUBE   0
#define TASK_TYPE_VECTOR 1
#define TASK_TYPE_MIX    2

#define TASK_STATE_CREATING  0
#define TASK_STATE_COMPLETED 1

#define SLOT_IDLE    0
#define SLOT_CLAIMED 1
#define SLOT_DONE    2

struct alignas(64) SimtAtomicLine {
    volatile int64_t value;
    uint8_t padding[56];
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
#ifdef __cplusplus
static_assert(sizeof(struct GmTaskDesc) == 128, "GmTaskDesc must be 128 bytes");
#else
_Static_assert(sizeof(struct GmTaskDesc) == 128, "GmTaskDesc must be 128 bytes");
#endif

struct GmPredecessorList {
    uint32_t cnt;
    uint32_t exp_offset;
};

struct alignas(64) GmSuccessorList {
    uint32_t cnt;
    uint32_t node[SIMT_CASE_CON_NODE_CNT];
    uint32_t next_offset;
    uint32_t pad;
};

struct GmTaskState {
    uint32_t state;
    uint32_t successor_cnt;
};

struct alignas(64) GmQueue {
    volatile uint64_t head;
    volatile uint64_t tail;
    volatile uint64_t cnt;
    uint32_t pad0[13];
    uint32_t tasks[SIMT_CASE_RING_SIZE];
    volatile int64_t lock;
    uint8_t pad1[24];
};

struct alignas(64) GmCtrl {
    uint64_t free_bitmap[SIMT_CASE_TASK_TYPE_CNT][SIMT_CASE_AIC_OSTD];
    uint64_t msg_bitmap[SIMT_CASE_EXE_TYPE_CNT][SIMT_CASE_AIC_OSTD];
    uint32_t task_id_map1[SIMT_CASE_EXE_TYPE_CNT][SIMT_CASE_AIC_CNT];
    uint32_t task_id_map2[SIMT_CASE_EXE_TYPE_CNT][SIMT_CASE_AIC_CNT];
};

struct SimtCaseState {
    SimtAtomicLine phase;
    SimtAtomicLine alloc_done;
    SimtAtomicLine desc_done;
    SimtAtomicLine all_done;
    SimtAtomicLine completed_cnt;
    SimtAtomicLine desc_commit_cursor;
    SimtAtomicLine pred_ring_cursor;

    uint32_t total_task_cnt;
    uint32_t pad0[7];

    struct GmTaskDesc basic_buf[SIMT_CASE_RING_SIZE];
    uint32_t pred_ring[SIMT_CASE_NODE_BUFF_SIZE];
    struct GmPredecessorList predecessors[SIMT_CASE_RING_SIZE];
    struct GmSuccessorList successor_buf[SIMT_CASE_RING_SIZE];
    struct GmTaskState state_buf[SIMT_CASE_RING_SIZE];
    uint32_t predecessor_cnt[SIMT_CASE_RING_SIZE];

    GmQueue ready_queue[2];
    GmQueue completed_queue;

    GmCtrl ctrl;

    SimtAtomicLine executor_done[64];

    uint64_t report_magic;
    uint64_t report_desc_writes;
    uint64_t report_cutter_ready;
    uint64_t report_dispatch_sent;
    uint64_t report_executor_done;
    uint64_t report_kernel_start_tick;
    uint64_t report_kernel_end_tick;
    uint64_t report_pad[8];

    uint64_t workspace_base;
    uint64_t workspace_bytes;
    uint64_t workspace_pad[6];
};

#ifdef __cplusplus
}
#endif

#endif
