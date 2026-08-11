# SIMT Case 设计方案：AICPU alloc + SIMT desc + SIMT cutter/dispatch + AICore executor

> 目标：在 A5（Ascend950）上实现 Qwen3-14B decoder 的完整 orchestration→scheduler 流水线
> 参考：
> - `esl_proxy/esl_proxy/cases/qwen3_14b_decoder_alloc.h` + `qwen3_14b_decoder_desc_ap.h`
> - `esl_proxy/esl_proxy/src/algorithm/cutter.c` + `dispatch.c` + `executor.c`
> - `simpler/tests/atomic_probe/pa_scheduler/simt_cross_core_v2`（SIMT 编程范式）

---

## 目录

- [1. 总体架构](#1-总体架构)
- [2. 硬件资源分配](#2-硬件资源分配)
- [3. GM 内存布局](#3-gm-内存布局)
- [4. 跨核通信协议](#4-跨核通信协议)
- [5. 各阶段详细设计](#5-各阶段详细设计)
- [6. 文件结构](#6-文件结构)
- [7. 编译方案](#7-编译方案)
- [8. 运行方案](#8-运行方案)
- [9. 环境设置](#9-环境设置)
- [10. 验证方案](#10-验证方案)

---

## 1. 总体架构

### 1.1 流水线全景

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        A5 单次 Kernel Launch                                 │
│                                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐         │
│  │  AICPU Host  │    │  AIV0 (SIMT)     │    │  AIC0..AIC31 +      │         │
│  │  (ACL Host)  │    │  desc + cutter   │    │  AIV2..AIV63        │         │
│  │              │    │  + dispatch      │    │  (executor cores)   │         │
│  │  alloc 阶段  │───▶│                  │───▶│                     │         │
│  │  初始化 GM   │    │  warp 0: cutter  │    │  poll→claim→done    │         │
│  │  启动 kernel │    │  warp 1: dispatch│    │  立即完成            │         │
│  │  等待完成    │    │  warp 2..N: desc │    │                     │         │
│  │  读回结果    │    │                  │    │                     │         │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘         │
│         │                     │  GM atomics                    │            │
│         │                     └────────────────────────────────┘            │
│         │                                                                   │
│         └──── D2H memcpy ── 验证结果                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 阶段划分

| 阶段 | 执行者 | 功能 | 参考源码 |
|------|--------|------|----------|
| **alloc** | AICPU（ACL Host） | 初始化 ext_* tensor 地址、分配中间 tensor、准备 GM 共享区域 | `qwen3_14b_decoder_alloc.h` |
| **desc** | AIV0 SIMT（2-8 warp） | 遍历 Qwen3 decode 图，写 g_basic_buf / g_predecessors / g_task_tensor_buf | `qwen3_14b_decoder_desc_ap.h` |
| **cutter** | AIV0 SIMT（warp 0） | 消费 g_predecessors 建 successor 链，无前驱 task 入 ready_queue | `cutter.c` |
| **dispatch** | AIV0 SIMT（warp 1） | 从 ready_queue 取 task，派发到 executor core（bitmap 协议） | `dispatch.c` |
| **executor** | AIC0..31 + AIV2..63 | 拿到任务后立即 CAS DONE（模拟零延迟执行） | `executor.c`（简化） |
| **回流** | AIV0 warp 0 | 读 completed_queue → resolve_dep → 新 ready task 入队 | `cutter.c` |

### 1.3 核心设计原则

1. **单次 Kernel Launch**：AICPU 通过 `aclrtLaunchKernelWithHostArgs(function_, 32U, ...)` 一次性启动 32 blocks（32 AIC + 64 AIV），所有阶段在同一 kernel 内完成
2. **GM 共享内存通信**：所有跨核数据交换通过 GM 原子操作（CAS / atomicAdd）+ reader DCCI
3. **Warp 角色分工**：AIV0 的不同 warp 分别承担 desc / cutter / dispatch 角色，可配置 4/8 warp
4. **立即完成执行器**：executor core 收到 task 后直接 CAS `CLAIMED→DONE`，验证调度正确性

---

## 2. 硬件资源分配

### 2.1 A5 拓扑

```
A5 = 32 blocks × (1 AIC + 2 AIV) = 32 AIC + 64 AIV
```

### 2.2 角色分配

| 角色 | 物理核 | 数量 | 说明 |
|------|--------|------|------|
| **SIMT 调度核**（desc+cutter+dispatch） | AIV0 block0 | 1 个 AIV | 用 SIMT VF 多 warp 并行 |
| **Executor CUBE** | AIC0..AIC31（block0 的 AIC 除外） | 31 个 | 模拟 Cube 立即完成 |
| **Executor VECTOR** | AIV2..AIV63（block0 的 AIV1 除外） | 62 个 | 模拟 Vector 立即完成 |
| **AICPU Host** | Host CPU | 1 | ACL 运行时，alloc + launch + D2H |

> Block0 的 AIC 留空或做 executor；Block0 的 AIV0 = 调度核，AIV1 = executor 或空闲。

### 2.3 Warp 配置

AIV0 SIMT VF 配置（编译时 `LAUNCH_BOUND` 控制）：

| 配置 | 总 warp | desc warp | cutter warp | dispatch warp | 总线程 |
|------|---------|-----------|-------------|---------------|--------|
| 4-warp | 4 | 2 | 1 | 1 | 128 |
| 8-warp | 8 | 4 | 2 | 2 | 256 |

```cpp
// 编译时宏
#ifndef SIMT_CASE_WARP_COUNT
#define SIMT_CASE_WARP_COUNT 8
#endif
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpCount = SIMT_CASE_WARP_COUNT;
constexpr uint32_t kThreadCount = kWarpCount * kWarpSize;

// 角色划分
constexpr uint32_t kDescWarpCount = kWarpCount / 2;      // 一半 warp 做 desc
constexpr uint32_t kCutterWarpCount = kWarpCount / 4;    // 1/4 做 cutter
constexpr uint32_t kDispatchWarpCount = kWarpCount / 4;  // 1/4 做 dispatch
```

---

## 3. GM 内存布局

### 3.1 总体布局

```
GM Base (aclrtMalloc)
│
├── [0x0000_0000] SimtCaseState           // 全局控制结构（~1MB）
│   ├── control                           // 启动配置 + drain 同步
│   ├── g_basic_buf[RING_SIZE]            // task_desc 数组
│   ├── g_predecessors_ring               // 前驱边扁平存储
│   ├── g_predecessors[RING_SIZE]         // 前驱列表（cnt + exp 指针）
│   ├── g_successor_buf[RING_SIZE]        // successor 链
│   ├── g_state_buf[RING_SIZE]            // task 生命周期状态
│   ├── g_predecessor_cnt[RING_SIZE]      // 未完成前驱计数
│   ├── ready_queue[2]                    // CUBE / VECTOR 就绪队列
│   ├── completed_queue                   // 完成 task 队列
│   ├── ctrl_t                            // dispatch 控制位图
│   ├── tensor_pool                       // 中间 tensor buffer 池
│   └── report                            // 诊断/统计报告区
│
├── [0x0100_0000] ext_weights             // 外部权重（模拟地址）
│   ├── ext_wq [5120×5120×2B = 50MB]
│   ├── ext_wk [5120×1024×2B = 10MB]
│   └── ...
│
└── [0x1000_0000] intermediate tensors    // 中间 tensor
    ├── q_proj, k_proj, v_proj, ...
    └── (alloc 阶段 bump-allocate)
```

### 3.2 核心数据结构（GM 对齐）

```cpp
// 64B cache-line 对齐的原子控制行
struct alignas(64) AtomicLine {
    volatile int64_t value;
    uint8_t padding[56];
};

// task_desc（简化版，适配 GM）
struct alignas(64) GmTaskDesc {
    uint32_t id;             // task id
    uint32_t type;           // 0=CUBE, 1=VECTOR, 2=MIX
    uint32_t count;          // SPMD block count
    uint32_t duration;       // 估计执行时长（ns / 10000）
    uint32_t tensor_cnt;     // data[] 有效条目数
    uint32_t scalar_cnt;     // scalar[] 有效条目数
    uint64_t data[16];       // tensor 地址
    int64_t  scalar[8];      // scalar 参数（精简）
};  // 256B = 4 cache lines

// 前驱列表
struct GmPredecessorList {
    uint32_t cnt;            // 前驱数
    uint32_t pad;
    uint32_t exp_offset;     // 在 g_predecessors_ring 中的偏移
};

// successor 链
struct alignas(64) GmSuccessorList {
    uint32_t cnt;
    uint32_t node[30];       // 最多 30 个直接 successor
    uint32_t next_offset;    // 溢出链
};

// task 状态
struct GmTaskState {
    uint32_t state;          // CREATING / COMPLETED
    uint32_t successor_cnt;
};

// 就绪队列（lock-free SPSC，cutter 写 / dispatch 读）
struct alignas(64) GmQueue {
    volatile uint64_t head;
    volatile uint64_t tail;
    volatile uint64_t cnt;
    uint32_t tasks[RING_SIZE];  // 4096 entries
    AtomicLine lock;             // spinlock
};

// dispatch 控制
struct alignas(64) GmCtrl {
    uint64_t free_bitmap[3][2];   // [type][slot] 空闲 AICore 位图
    uint64_t msg_bitmap[2][2];    // [exe_type][slot] 完成通知位图
    uint32_t task_id_map1[2][64]; // [exe_type][core] → task_id (slot 0)
    uint32_t task_id_map2[2][64]; // slot 1
};

// 全局状态
struct alignas(4096) SimtCaseState {
    // 控制区
    AtomicLine phase;             // 0=alloc, 1=desc, 2=schedule, 3=done
    AtomicLine alloc_done;        // AICPU 通知 SIMT: alloc 完成
    AtomicLine desc_done;         // SIMT desc 通知 cutter: desc 完成
    AtomicLine all_done;          // 全部完成
    uint32_t total_task_cnt;      // 总 task 数
    uint32_t completed_cnt;       // 已完成 task 数（atomic）
    
    // DAG 数据
    GmTaskDesc g_basic_buf[RING_SIZE];           // 4096 × 256B = 1MB
    uint32_t g_predecessors_ring[NODE_BUFF_SIZE]; // 65536 × 4B = 256KB
    GmPredecessorList g_predecessors[RING_SIZE];  // 4096 × 8B = 32KB
    GmSuccessorList g_successor_buf[RING_SIZE];    // 4096 × 64B = 256KB
    GmTaskState g_state_buf[RING_SIZE];            // 4096 × 8B = 32KB
    uint32_t g_predecessor_cnt[RING_SIZE];         // 4096 × 4B = 16KB
    
    // 队列
    GmQueue ready_queue[2];    // [0]=CUBE, [1]=VECTOR
    GmQueue completed_queue;   // executor → cutter
    
    // dispatch 控制
    GmCtrl ctrl;
    
    // 报告
    AtomicLine desc_warp_report[8];   // 每 warp 的 desc 统计
    AtomicLine cutter_report;
    AtomicLine dispatch_report;
    uint64_t kernel_start_ticks;
    uint64_t kernel_end_ticks;
};
```

### 3.3 内存大小估算

| 区域 | 大小 | 说明 |
|------|------|------|
| `SimtCaseState` | ~2MB | 全部控制 + DAG 数据 |
| `ext_weights` | ~256MB | 权重（模拟，不实际访问） |
| `intermediate` | ~512MB | 中间 tensor（模拟） |
| **总计** | ~770MB | `aclrtMalloc` 分配 |

---

## 4. 跨核通信协议

### 4.1 阶段同步

```
AICPU Host:
  1. alloc GM → 初始化 SimtCaseState
  2. atomicStore(phase, 1) → 通知 SIMT 开始 desc
  3. 等待 atomicLoad(all_done) == 1

AIV0 SIMT (desc warps):
  1. 等待 atomicLoad(phase) >= 1
  2. warp-interleaved 写 g_basic_buf + g_predecessors
  3. 最后一个 desc warp: atomicStore(desc_done, 1)

AIV0 SIMT (cutter warp):
  1. 等待 atomicLoad(desc_done) == 1
  2. 消费 g_predecessors → ready_queue
  3. 循环处理 completed_queue → resolve_dep

AIV0 SIMT (dispatch warp):
  1. 等待 ready_queue 有 task
  2. 派发到 executor core (CAS)
  3. 读 msg_bitmap → completed_queue

AIC/AIV Executor:
  1. CAS CLAIM task
  2. 立即 CAS DONE
  3. 写 msg_bitmap
```

### 4.2 Task 生命周期

```
                  desc warp               cutter warp          dispatch warp        executor core
                    │                         │                     │                    │
  g_basic_buf[i]    │── write desc ──▶        │                     │                    │
  g_predecessors[i] │── write edges ──▶       │                     │                    │
                    │                         │                     │                    │
  g_state_buf[i]    │                   ── CREATING ──▶             │                    │
  g_predecessor_cnt │                   ── =N ──▶                   │                    │
  ready_queue       │                   ── enqueue (if cnt==0) ──▶  │                    │
                    │                         │                     │                    │
  task slot         │                         │                ── CAS CLAIM ──▶          │
                    │                         │                     │              ── CAS DONE ──
  msg_bitmap        │                         │                     │ ◀── write bit ────│
  completed_queue   │                         │ ◀── enqueue ────────│                    │
  g_state_buf[i]    │                   ── COMPLETED ──▶            │                    │
  resolve_dep       │                   ── decrement successor ──▶  │                    │
```

### 4.3 原子操作清单

| 操作 | 指令 | 用途 |
|------|------|------|
| 阶段同步 | `asc_atomic_cas` / `atomicCAS` | phase / desc_done / all_done 状态机 |
| Queue 加锁 | `asc_atomic_cas` | GmQueue.spinlock（cutter/dispatch 互斥） |
| Task 派发 | `asc_atomic_cas` | task slot: IDLE→CLAIMED→DONE |
| 完成通知 | `asc_atomic_or` | msg_bitmap 置位（executor → dispatch） |
| 计数器 | `asc_atomic_add` | completed_cnt / g_predecessor_cnt 递减 |
| DCCI | `asc_dcci_single` + `dsb` | reader 端清 cache line（payload 可见性） |

### 4.4 DCCI 可见性规则

参考 `simt_cross_core_v2` 的实验结论：

> **SIMT/Scalar 普通 GM store/load 不共享统一 DCache。claim winner 必须 `dcci + dsb` 每个 payload cache line 后才能普通读。**

本方案中：
- **desc → cutter**：desc warp 写 `g_basic_buf` / `g_predecessors`，cutter warp 读。cutter 在 `desc_done` CAS 成功后，对 `g_basic_buf[task_id]` 做 `dcci + dsb` 再读。
- **dispatch → executor**：dispatch warp 写 task slot payload，executor 做 `dcci + dsb` 后读。
- **executor → dispatch**：executor 写 `msg_bitmap`（原子操作，不需要 DCCI），dispatch 用 `atomicAdd(0)` 轮询。
- **dispatch → cutter**：dispatch 写 `completed_queue`（加锁），cutter 读（加锁），lock/unlock 的 acquire/release 语义保证可见性。

---

## 5. 各阶段详细设计

### 5.1 alloc 阶段（AICPU Host）

**位置**：`simt_case/host/alloc_host.cpp`

**功能**：
1. `aclrtMalloc` 分配 GM（SimtCaseState + tensor pool）
2. 初始化 20 个 `ext_*` tensor 地址（指向 GM 中的模拟权重区）
3. Bump-allocate 中间 tensor 到 `tensor_pool`（与 `alloc.h` 逻辑相同）
4. 初始化 `SimtCaseState`：清零所有队列、位图、状态
5. `atomicStore(phase, 1)` 通知 SIMT 开始
6. `aclrtLaunchKernelWithHostArgs` 启动 kernel
7. 等待 `all_done`，`aclrtMemcpy D2H` 读回报告

```cpp
// Host 伪码
void alloc_phase(SimtCaseState* gm_state) {
    // 初始化 ext_* 地址（模拟，指向 GM 权重区）
    uint64_t weights_base = gm_state + offsetof(SimtCaseState, tensor_pool);
    ext_wq_addr = weights_base + 0;
    ext_wk_addr = weights_base + 50*1024*1024;
    // ...
    
    // Bump-allocate 中间 tensor
    uint64_t pool_base = weights_base + WEIGHTS_TOTAL;
    // 与 alloc.h 相同的循环结构
    for (b0 = 0; b0 < 96; b0 += 16) {
        alloc_tensor(pool_base, {16, 5120}, BFLOAT16);  // normed_tile
        alloc_task_id++;
        for (base = 0; base < 20; base += bpt) alloc_task_id++;  // Q-Proj
        // ...
    }
    
    gm_state->total_task_cnt = alloc_task_id;
    atomicStore(&gm_state->phase, 1);
}
```

### 5.2 desc 阶段（AIV0 SIMT, 多 warp）

**位置**：`simt_case/kernel/desc_simt.cpp`

**功能**：Warp-interleaved 遍历 Qwen3 decode 图，写 `g_basic_buf` + `g_predecessors`。

**Warp 划分**（striped，与 `DESC_DO_OR_SKIP` 相同策略）：
```cpp
// warp w 拥有 task_id ∈ [w, w+kDescWarpCount, w+2*kDescWarpCount, ...]
const uint32_t warp = threadIdx.x / 32;
const uint32_t lane = threadIdx.x % 32;
const bool is_leader = (lane == 0);

if (warp >= kDescWarpCount) return;  // 非 desc warp 退出

// Striped task 归属
for (uint32_t task_seq = warp; task_seq < total_task_cnt; task_seq += kDescWarpCount) {
    // 按 desc_ap.h 的循环结构确定当前 task_seq 对应哪个算子
    // 写 g_basic_buf[task_seq].{type, count, duration, data[], scalar[]}
    // 写 g_predecessors[task_seq]（显式前驱）
}
```

**关键简化**：
- `g_basic_buf` 用简化版 `GmTaskDesc`（256B 而非 432B，去掉 kernel 指针等字段）
- `g_predecessors_ring` 改为 GM 中的固定偏移数组，用 `atomicAdd` 分配 offset
- tensor 地址用 alloc 阶段的 bump offset（不需要真实 tensor 数据）

**Desc 完成同步**：
```cpp
// 最后一个 desc warp 通知 cutter
uint64_t prev = asc_atomic_add(&desc_done_counter, 1);
if (prev + 1 == kDescWarpCount) {
    asc_threadfence();
    asc_atomic_cas(&gm_state->desc_done, 0, 1);
}
```

### 5.3 cutter 阶段（AIV0 SIMT, warp 0）

**位置**：`simt_case/kernel/cutter_simt.cpp`

**功能**：与 `cutter.c` 逻辑相同，适配 GM + SIMT。

```cpp
// cutter warp = kDescWarpCount（第一个非 desc warp）
if (warp != kCutterWarpId) return;  // 只有 cutter warp 执行
if (!is_leader) return;              // lane 0 执行

while (true) {
    // (1) 提交新 task：消费 g_predecessors
    uint32_t commit_end = min(g_commit_task_id + PRE_BATCH_SIZE, total_task_cnt);
    while (g_commit_task_id < commit_end) {
        GmPredecessorList* pl = &gm_state->g_predecessors[g_commit_task_id];
        if (pl->cnt == 0) {
            // 无前驱 → ready
            uint32_t type = gm_state->g_basic_buf[g_commit_task_id].type;
            enqueue(&gm_state->ready_queue[type], g_commit_task_id);
        } else {
            // 有前驱 → 建 successor 链
            uint32_t pending = 0;
            for (i = 0; i < pl->cnt; i++) {
                uint32_t pred = gm_state->g_predecessors_ring[pl->exp_offset + i];
                if (gm_state->g_state_buf[pred].state != COMPLETED) {
                    // 追加 successor
                    uint32_t sidx = gm_state->g_successor_buf[pred].cnt++;
                    gm_state->g_successor_buf[pred].node[sidx] = g_commit_task_id;
                    gm_state->g_state_buf[pred].successor_cnt++;
                    pending++;
                }
            }
            gm_state->g_predecessor_cnt[g_commit_task_id] = pending;
            if (pending == 0) {
                uint32_t type = gm_state->g_basic_buf[g_commit_task_id].type;
                enqueue(&gm_state->ready_queue[type], g_commit_task_id);
            }
        }
        g_commit_task_id++;
    }
    
    // (2) 处理完成队列
    uint32_t cq_buf[CQ_BATCH_SIZE];
    uint32_t cnt = CQ_BATCH_SIZE;
    if (batch_dequeue(&gm_state->completed_queue, cq_buf, &cnt)) {
        for (j = 0; j < cnt; j++) {
            uint32_t tid = cq_buf[j];
            gm_state->g_state_buf[tid].state = COMPLETED;
            
            // resolve_dep: 遍历 successor
            uint32_t sc = gm_state->g_state_buf[tid].successor_cnt;
            for (k = 0; k < sc; k++) {
                uint32_t succ = gm_state->g_successor_buf[tid].node[k];
                uint32_t prev = asc_atomic_add(&gm_state->g_predecessor_cnt[succ], -1);
                if (prev == 1) {  // 归零 → ready
                    uint32_t type = gm_state->g_basic_buf[succ].type;
                    enqueue(&gm_state->ready_queue[type], succ);
                }
            }
            
            asc_atomic_add(&gm_state->completed_cnt, 1);
        }
    }
    
    // (3) 检查终止
    if (atomicLoad(&gm_state->completed_cnt) >= total_task_cnt)
        break;
}
```

### 5.4 dispatch 阶段（AIV0 SIMT, warp 1）

**位置**：`simt_case/kernel/dispatch_simt.cpp`

**功能**：与 `dispatch.c` 逻辑相同，适配 GM + SIMT。

```cpp
// dispatch warp = kDescWarpCount + 1
if (warp != kDispatchWarpId) return;
if (!is_leader) return;

while (true) {
    // (1) 读完成通知：msg_bitmap → free_bitmap → completed_queue
    for (type = 0; type < 2; type++) {
        for (slot = 0; slot < 2; slot++) {
            uint64_t msg = gm_state->ctrl.msg_bitmap[type][slot];
            if (msg) {
                // 解码完成的 task
                while (msg) {
                    uint64_t idx = __builtin_ctzll(msg);
                    uint32_t tid = (slot == 0) 
                        ? gm_state->ctrl.task_id_map1[type][idx]
                        : gm_state->ctrl.task_id_map2[type][idx];
                    enqueue(&gm_state->completed_queue, tid);
                    // 恢复 free
                    gm_state->ctrl.free_bitmap[type][slot] |= (1ULL << idx);
                    msg &= msg - 1;
                }
                gm_state->ctrl.msg_bitmap[type][slot] = 0;  // 清除
            }
        }
    }
    
    // (2) 派发：ready_queue → executor slot
    for (type = 0; type < 2; type++) {
        uint64_t free = gm_state->ctrl.free_bitmap[type][0] 
                      & gm_state->ctrl.free_bitmap[type][1];
        int free_cnt = __builtin_popcountll(free);
        if (free_cnt <= 0) continue;
        
        uint32_t task_ids[64];
        uint32_t got = free_cnt;
        if (!batch_dequeue(&gm_state->ready_queue[type], task_ids, &got))
            continue;
        
        for (i = 0; i < got; i++) {
            uint64_t idx = __builtin_ctzll(free);
            uint64_t mask = 1ULL << idx;
            int slot = (gm_state->ctrl.free_bitmap[type][0] & mask) ? 0 : 1;
            
            // 写 task slot（executor 会 CAS 这个地址）
            uint32_t core_base = (type == 0) ? kCubeCoreBase : kVecCoreBase;
            uint64_t slot_addr = executor_slot_addr(core_base + idx, slot);
            
            // DCCI + write payload + DCCI + CAS publish
            dcci(slot_addr);  // 清旧 cache
            *slot_addr = task_ids[i];  // 写 task_id
            dcci(slot_addr);  // 刷新新 cache
            dsb();
            asc_atomic_cas(slot_state_addr(core_base + idx, slot), IDLE, CLAIMED);
            
            // 记录映射
            if (slot == 0)
                gm_state->ctrl.task_id_map1[type][idx] = task_ids[i];
            else
                gm_state->ctrl.task_id_map2[type][idx] = task_ids[i];
            
            gm_state->ctrl.free_bitmap[type][slot] &= ~mask;
            free &= ~mask;
        }
    }
    
    // (3) 检查终止
    if (atomicLoad(&gm_state->completed_cnt) >= total_task_cnt)
        break;
}
```

### 5.5 executor 阶段（AIC / AIV Scalar）

**位置**：`simt_case/kernel/executor_scalar.cpp`

**功能**：轮询自己的 task slot，拿到 task 后立即 CAS DONE。

```cpp
// 每个 executor core 的 Scalar 入口
__aicore__ void executor_entry(__gm__ SimtCaseState* state) {
    const uint32_t core_id = get_coreid();
    const uint32_t exe_type = is_cube_core(core_id) ? 0 : 1;  // 0=CUBE, 1=VECTOR
    const uint32_t local_core = core_id - core_base(exe_type);
    
    // 每个 core 有 2 个 slot（PING-PONG）
    while (true) {
        for (slot = 0; slot < 2; slot++) {
            uint64_t* slot_state = slot_state_addr(core_id, slot);
            
            // 轮询 CLAIMED 状态
            uint64_t s = atomicAdd(slot_state, 0);  // atomic load
            if (s == CLAIMED) {
                // CAS: CLAIMED → DONE（立即完成）
                if (atomicCAS(slot_state, CLAIMED, DONE) == CLAIMED) {
                    // 写 msg_bitmap 通知 dispatch
                    uint64_t mask = 1ULL << local_core;
                    atomicOr(&state->ctrl.msg_bitmap[exe_type][slot], mask);
                }
            }
        }
        
        // 检查全局完成
        if (atomicAdd(&state->completed_cnt, 0) >= state->total_task_cnt)
            break;
    }
}
```

### 5.6 Kernel 主入口

```cpp
// 混合 ELF 入口：AIC + AIV
__aicore__ void simt_case_main(__gm__ uint64_t* state_ptr) {
    __gm__ SimtCaseState* state = (__gm__ SimtCaseState*)state_ptr;
    
    uint32_t core_type = get_core_type();  // AIC or AIV
    uint32_t core_id = get_coreid();
    
    if (core_id == kSchedulerCoreId) {
        // AIV0: SIMT 调度核
        // 启动 SIMT VF: desc + cutter + dispatch
        cce::async_invoke<simt_scheduler_vf>(
            cce::dim3{kThreadCount, 1, 1},
            state
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    } else {
        // AIC0..31 + AIV2..63: executor
        executor_entry(state);
    }
    
    // drain 同步
    atomicAdd(&state->drain_arrivals, 1);
    while (atomicAdd(&state->drain_arrivals, 0) < kTotalCores) {
        // spin
    }
}
```

---

## 6. 文件结构

```
simt_case/
├── design.md                          # 本设计文档
├── run.sh                             # 统一编译/运行入口
├── common/
│   ├── simt_case_protocol.h           # GM 数据结构定义（GmTaskDesc / GmQueue / GmCtrl 等）
│   ├── simt_case_model.h              # Qwen3 decode 图模型（task 序列定义、SPMD 切分）
│   └── simt_case_config.h             # 编译时配置（warp count / RING_SIZE / 核数等）
├── kernel/
│   ├── simt_case_kernel.cpp           # 主入口（AIC + AIV 混合 ELF）
│   ├── desc_simt.cpp                  # SIMT desc 阶段（VF 函数）
│   ├── cutter_simt.cpp                # SIMT cutter 阶段（VF 函数）
│   ├── dispatch_simt.cpp              # SIMT dispatch 阶段（VF 函数）
│   └── executor_scalar.cpp            # executor Scalar 逻辑
├── host/
│   ├── alloc_host.cpp                 # AICPU alloc 阶段 + ACL launch
│   └── verify_host.cpp                # D2H 验证逻辑
├── cpu/                               # CPU 模拟版（用于先验调试）
│   ├── cpu_model.cpp                  # 完整流水线 CPU 仿真
│   └── build.sh
├── ccec/
│   ├── build.sh                       # CCEC 编译脚本（AIC + AIV → 混合 ELF）
│   └── simt_stack_acl.json            # SIMT stack 配置
├── test/
│   └── test_simt_case.cpp             # CPU 侧单元测试
└── README.md                          # 使用说明
```

---

## 7. 编译方案

### 7.1 三层编译（参考 `simt_cross_core_v2`）

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: CPU 模拟版                                     │
│   g++ -O2 -std=c++17 -fsanitize=address,undefined       │
│   → cpu/simt_case_cpu                                   │
│   用途：先验调试数据结构和算法逻辑                       │
└──────────────────────┬──────────────────────────────────┘
                       │ 逻辑验证通过后
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: CCEC 混合 ELF                                  │
│   ccec --cce-aicore-arch=dav-c310-cube  → AIC .o        │
│   ccec --cce-aicore-arch=dav-c310-vec   → AIV .o (SIMT) │
│   ld.lld -m aicorelinux → mixed ELF                     │
│   用途：A5 上板运行                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3: ACL Host                                       │
│   g++-15 -std=c++17 -lascendcl -lruntime               │
│   → host/simt_case_host                                 │
│   用途：AICPU alloc + launch + D2H 验证                 │
└─────────────────────────────────────────────────────────┘
```

### 7.2 CCEC 编译脚本（`ccec/build.sh`）

```bash
#!/bin/bash
set -euo pipefail

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH not set}"
CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"

WARP_COUNT="${SIMT_CASE_WARP_COUNT:-8}"
RING_SIZE="${SIMT_CASE_RING_SIZE:-4096}"

COMMON_FLAGS=(
    -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -DSIMT_CASE_WARP_COUNT=$WARP_COUNT
    -DSIMT_CASE_RING_SIZE=$RING_SIZE
    -I../common
)

SRC="../kernel/simt_case_kernel.cpp"

# AIC (Cube) 编译
"$CCEC" -c "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-cube \
    -o build/aic.o "$SRC"

# AIV (Vector + SIMT) 编译
"$CCEC" -c "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -o build/aiv.o "$SRC"

# 链接混合 ELF
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o build/simt_case.o build/aic.o build/aiv.o

echo "Built: build/simt_case.o (warp=$WARP_COUNT, ring=$RING_SIZE)"
```

### 7.3 Host 编译脚本

```bash
#!/bin/bash
set -euo pipefail

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH not set}"
GXX15="${GXX15:-g++-15}"

WARP_COUNT="${SIMT_CASE_WARP_COUNT:-8}"

"$GXX15" -O2 -std=c++17 -Wall -Wextra \
    -DSIMT_CASE_WARP_COUNT=$WARP_COUNT \
    -I../common \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    ../host/alloc_host.cpp \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl \
    -o build/simt_case_host

echo "Built: build/simt_case_host (warp=$WARP_COUNT)"
```

---

## 8. 运行方案

### 8.1 统一入口（`run.sh`）

```bash
#!/bin/bash
set -euo pipefail

# 子命令
case "${1:-}" in
    build-cpu)   # CPU 模拟版
        cd cpu && ./build.sh
        ;;
    run-cpu)     # 运行 CPU 模拟
        ./build/simt_case_cpu --batches 256 --runs 10
        ;;
    build-ccec)  # CCEC 混合 ELF
        cd ccec && ./build.sh
        ;;
    build-host)  # ACL Host
        cd host && ./build.sh
        ;;
    build-all)   # 全部编译
        ./run.sh build-cpu
        ./run.sh build-ccec
        ./run.sh build-host
        ;;
    run-board)   # 上板运行
        DEVICE="${2:-0}"
        ./build/simt_case_host --device $DEVICE --batches 256 --runs 10
        ;;
    run-board-task)  # 通过 task-submit 运行
        task-submit --device 0 --timeout 600 --run \
            "cd simt_case && ./run.sh run-board \$TASK_DEVICE"
        ;;
    *)
        echo "Usage: $0 {build-cpu|run-cpu|build-ccec|build-host|build-all|run-board|run-board-task}"
        echo "  Env: SIMT_CASE_WARP_COUNT=4|8  (default 8)"
        echo "       SIMT_CASE_RING_SIZE=4096   (default)"
        exit 1
        ;;
esac
```

### 8.2 典型运行流程

```bash
# 1. 环境设置
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="/tmp/opencode/localtools/bin:$PATH"
echo 'a5|Ascend950PR_9579|Ascend950' > /tmp/onboard-arch-precheck.cache

# 2. CPU 先验调试（4 warp）
SIMT_CASE_WARP_COUNT=4 ./run.sh build-cpu
SIMT_CASE_WARP_COUNT=4 ./run.sh run-cpu

# 3. 上板编译（8 warp）
SIMT_CASE_WARP_COUNT=8 ./run.sh build-ccec
SIMT_CASE_WARP_COUNT=8 ./run.sh build-host

# 4. 上板运行
SIMT_CASE_WARP_COUNT=8 ./run.sh run-board 0

# 5. 通过 task-submit 运行（受限环境）
./run.sh run-board-task

# 6. 对比 4 warp vs 8 warp
SIMT_CASE_WARP_COUNT=4 ./run.sh build-ccec && ./run.sh run-board 0
SIMT_CASE_WARP_COUNT=8 ./run.sh build-ccec && ./run.sh run-board 0
```

### 8.3 预期输出

```
[ALLOC] device=0 soc=Ascend950PR_9579 topology=32*(1AIC+2AIV) warps=8
[ALLOC] total_task_cnt=3096 tensor_pool=512MB
[LAUNCH] blocks=32 function=simt_case_main
[KERNEL] start_ticks=0x12345678
[DESC] warps=4 tasks=3096 elapsed=1.2ms
[CUTTER] committed=3096 ready=128 resolved=2968
[DISPATCH] sent=3096 completed=3096
[EXECUTOR] cores=93 (31 cube + 62 vec) immediate_complete=yes
[DONE] all_done=1 completed_cnt=3096/3096 kernel_us=2340
[VERIFY] PASS: all tasks completed, DAG edges resolved
[PERF] kernel_time=2.34ms throughput=1.32 MTasks/s
```

---

## 9. 环境设置

### 9.1 编译环境

| 项目 | 要求 | 设置方式 |
|------|------|----------|
| ASCEND_HOME_PATH | CANN 工具包路径 | `export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest` |
| ccec | CCEC 编译器 | `$ASCEND_HOME_PATH/bin/ccec` |
| ld.lld | AICore 链接器 | `$ASCEND_HOME_PATH/bin/ld.lld` |
| g++-15 | Host 编译器 | `export GXX15=g++-15` 或绝对路径 |
| ACL 库 | `libascendcl.so` | `$ASCEND_HOME_PATH/x86_64-linux/lib64` |

### 9.2 运行环境

| 项目 | 要求 | 设置方式 |
|------|------|----------|
| 设备 | A5 / Ascend950 | `npu-smi info` 确认 |
| 权限 | root 或 HwHiAiUser | `ls -l /dev/davinci0` |
| task-submit | 受限环境必需 | `which task-submit` |
| 架构预检缓存 | 跳过 onboarding | `echo 'a5\|Ascend950PR_9579\|Ascend950' > /tmp/onboard-arch-precheck.cache` |

### 9.3 SIMT Stack 配置

```json
// ccec/simt_stack_acl.json
{
    "StackSize": {
        "simt_stack_size": 1536,
        "simt_divergence_stack_size": 4608
    }
}
```

Host 初始化时传入：
```cpp
aclrtSetDevice(device);
aclInit("simt_case/ccec/simt_stack_acl.json");
```

---

## 10. 验证方案

### 10.1 三层验证

| 层 | 工具 | 验证内容 |
|----|------|----------|
| **CPU 模拟** | g++ + ASan + UBSan | 数据结构正确性、DAG 边完整性、队列溢出、死锁检测 |
| **CCEC 静态** | ccec + llvm-bcanalyzer | ELF 符号、SIMT VF 入口、bitcode intrinsic 合法性 |
| **A5 动态** | ACL launch + D2H | 100 轮同地址复用、task 全部完成、completed_cnt == total_task_cnt |

### 10.2 正确性检查项

1. **alloc**：`total_task_cnt` 与 desc_ap.h 的 task 数一致（tier 0 ≈ 3096）
2. **desc**：每个 `g_basic_buf[i].{type, count, duration}` 非零；`g_predecessors[i].cnt` 与 desc_ap.h 的 `add_predecessors` 调用次数一致
3. **cutter**：`g_commit_task_id == total_task_cnt`；`g_predecessor_cnt` 全部归零
4. **dispatch**：`msg_bitmap` 全清零；`task_id_map` 全部映射到已完成 task
5. **executor**：所有 task slot 状态为 `DONE`
6. **全局**：`completed_cnt == total_task_cnt`

### 10.3 性能指标

| 指标 | 采集方式 | 期望值 |
|------|----------|--------|
| kernel 总时间 | `aclrtEventElapsedTime` | < 5ms（3096 task） |
| desc 吞吐 | `kernel_time / total_task_cnt` | > 1 MTasks/s |
| 4-warp vs 8-warp | 对比 `kernel_us` | 8-warp 应更快 |
| 同地址复用稳定性 | 100 轮 PASS 率 | 100/100 |

### 10.4 DCCI 验证

参照 `simt_cross_core_v2` 的 4 模式对比：

| 模式 | 预期 | 说明 |
|------|------|------|
| NO_DCCI | FAIL | 普通 GM store/load 不可靠 |
| WRITER_DCCI | FAIL | writer 端 DCCI 无效 |
| READER_DCCI | PASS | reader 端 DCCI + DSB 可靠 |
| WRITER+READER | PASS | 冗余但正确 |

---

## 附录 A：Qwen3 Decode 图 Task 序列（tier 0, 3096 tasks）

| 段 | 算子 | 每 tile task 数 | tile 数 | 小计 |
|----|------|-----------------|---------|------|
| 1 | RMSNorm | 1 | 6 | 6 |
| 1 | Q-Proj | 20 | 6 | 120 |
| 1 | K-Proj | 8 | 6 | 48 |
| 1 | V-Proj | 8 | 6 | 48 |
| 1 | QK-Norm | 1 | 6 | 6 |
| 2 | ROPE | 1 | 90 | 90 |
| 2 | QK-Matmul | 4 | 90 | 360 |
| 2 | Softmax | 4 | 90 | 360 |
| 2 | SV-Matmul | 4 | 90 | 360 |
| 2 | Online-Softmax | 4 | 90 | 360 |
| 3 | Out-Proj | 40 | 6 | 240 |
| 3 | Post-RMSNorm | 1 | 6 | 6 |
| 3 | Gate-Proj | 34 | 6 | 204 |
| 3 | Up-Proj | 34 | 6 | 204 |
| 3 | SILU | 34 | 6 | 204 |
| 3 | Down-Proj | 40 | 6 | 240 |
| 3 | Down-Proj-Res | 40 | 6 | 240 |
| **总计** | | | | **3096** |

## 附录 B：关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `RING_SIZE` | 4096 | task ring buffer 容量 |
| `NODE_BUFF_SIZE` | 65536 | 前驱边扁平存储容量 |
| `CON_NODE_CNT` | 30 | 每节点内联 successor 上限 |
| `kWarpSize` | 32 | SIMT warp 大小 |
| `kWarpCount` | 4 / 8 | 可配置 warp 数 |
| `kSchedulerCoreId` | AIV0 block0 | 调度核物理 ID |
| `kCubeCoreBase` | AIC block0 | Cube executor 起始 |
| `kVecCoreBase` | AIV block1 | Vector executor 起始 |
| `kTotalExecutorCores` | 93 | 31 Cube + 62 Vector |
| `AIC_OSTD` | 2 | 每 core 双槽 PING-PONG |
