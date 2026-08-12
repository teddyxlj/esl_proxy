# SIMT Qwen3 Decode Case — 设计文档 v2

> AICPU alloc → SIMT VF (desc + cutter + dispatch) → AIC/AIV executor (CAS immediate-complete)
> 不依赖 PA 参考代码，使用自有 `simt_case_protocol.h` 数据结构。

---

## 1. 硬件资源分配

| 角色 | 物理核 | 数量 | 说明 |
|------|--------|------|------|
| **AICPU Host** | Host CPU | 1 | alloc 阶段：初始化 GM + 构建 Qwen3 task graph + launch + D2H 验证 |
| **AIV0 SIMT VF** | block0 subblock0 | 1 AIV | 一个 SIMT VF 内多 warp 分工：desc warps + cutter warp + dispatch warp |
| **AIC0..31 executor** | block0..31 AIC | 32 | Cube executor：轮询 exe_slots，CAS CLAIM→DONE 立即完成 |
| **AIV2..63 executor** | block1..31 subblock1 | 63 | Vector executor：同上 |

### Warp 配置（SC_WARP_COUNT=4 或 8）

| 配置 | desc warps | cutter warp | dispatch warp | 总线程 |
|------|-----------|-------------|---------------|--------|
| 4-warp | 2 | 1 | 1 | 128 |
| 8-warp | 4 | 2 | 2 | 256 |

```c
#define SC_DESC_WARPS    (SC_WARP_COUNT / 2)
#define SC_CUTTER_WARP   (SC_DESC_WARPS)
#define SC_DISPATCH_WARP (SC_DESC_WARPIS + 1)
```

---

## 2. GM 内存布局

```
ScState (single aclrtMalloc, ~2MB)
├── phase / desc_done / all_done / completed_cnt   (原子同步)
├── basic_buf[4096]         — GmTaskDesc (128B each = 512KB)
├── pred_ring[65536]        — 前驱边扁平存储 (256KB)
├── predecessors[4096]      — GmPredList (8B each = 32KB)
├── successor_buf[4096]     — GmSuccList (128B each = 512KB)
├── state_buf[4096]         — GmTaskState (8B each = 32KB)
├── pred_cnt[4096]          — uint32 (16KB)
├── ready_queue[2]          — CUBE / VECTOR 就绪队列
├── completed_queue         — executor → cutter 完成队列
├── exe_slots[186]          — 93 cores × 2 slots (CAS 协议)
├── free_bitmap / msg_bitmap — dispatch 位图
└── report fields           — 统计
```

---

## 3. 执行流程

```
AICPU Host:
  1. aclrtMalloc ScState
  2. 构建 Qwen3 task graph (tier 0, 3096 tasks, 27450 edges)
  3. 填充 basic_buf / predecessors / pred_ring / state_buf
  4. 设置 total_task_cnt, phase=DESC
  5. aclrtLaunchKernelWithHostArgs(32 blocks)
  6. 等待 all_done, D2H 读回 report

AIV0 (block0 subblock0) — SIMT VF:
  warp 0..N-1: desc warps (striped, 写 basic_buf + predecessors — 已由 host 预填, VF 只设 state=CREATING + desc_done)
  warp N: cutter (消费 predecessors → ready_queue, 处理 completed_queue → resolve_dep)
  warp N+1: dispatch (ready_queue → exe_slots CAS CLAIMED, 读 msg_bitmap → completed_queue)

AIC0..31 + AIV2..63 — executor:
  轮询自己的 exe_slots[core*2+slot]
  CAS CLAIMED → DONE 立即完成
  写 msg_bitmap 通知 dispatch
```

### 3.1 AICPU alloc 阶段

Host CPU 执行 `qwen3_14b_decoder_alloc.h` 的逻辑：
- 初始化 20 个 ext_* tensor 地址（模拟）
- Bump-allocate 中间 tensor
- **关键**：构建完整的 Qwen3 decode task graph（3096 tasks），写入 GM：
  - `basic_buf[i].{id, type, count, duration}` — 按 desc_ap.h 的循环顺序
  - `predecessors[i].{cnt, exp_offset}` + `pred_ring[]` — 显式前驱边
  - `state_buf[i].state = CREATING`

### 3.2 SIMT desc 阶段

desc warps 做 striped 遍历（与 desc_ap.h 的 `DESC_DO_OR_SKIP` 相同策略）：
- 每个 warp 处理 task_id ∈ [warp, warp+N, warp+2N, ...]
- 由于 host 已预填 basic_buf + predecessors，desc VF 只需：
  - 设置 `state_buf[task_id].state = CREATING`
  - 设置 `successor_buf[task_id].cnt = 0`
- 最后一个 desc warp 完成后设 `desc_done = 1`

### 3.3 SIMT cutter 阶段（warp N）

cutter warp 消费 `predecessors[]` 建 successor 链：
```
for i in 0..total:
  if predecessors[i].cnt == 0:
    enqueue(ready_queue[type[i]], i)
    pred_cnt[i] = 0
  else:
    pending = 0
    for j in predecessors[i]:
      if state_buf[pred].state != COMPLETED:
        successor_buf[pred].node[...] = i
        pending++
    pred_cnt[i] = pending
    if pending == 0: enqueue(ready_queue, i)

loop:
  dequeue(completed_queue) → for each task_id:
    state_buf[task_id] = COMPLETED
    for succ in successor_buf[task_id]:
      pred_cnt[succ]--
      if pred_cnt[succ] == 0: enqueue(ready_queue, succ)
  if completed_cnt >= total: break
```

### 3.4 SIMT dispatch 阶段（warp N+1）

dispatch warp 从 ready_queue 取 task，派发到 executor slots：
```
loop:
  // 1. 读完成: msg_bitmap → completed_queue
  for type in [CUBE, VECTOR]:
    for slot in [0, 1]:
      msg = msg_bitmap[type][slot]
      while msg:
        core = ctz(msg)
        task_id = exe_slots[core*2+slot].task_id
        enqueue(completed_queue, task_id)
        free_bitmap[type][slot] |= (1 << core)
        msg &= msg - 1
      msg_bitmap[type][slot] = 0

  // 2. 派发: ready_queue → exe_slots
  for type in [CUBE, VECTOR]:
    free = free_bitmap[type][0] & free_bitmap[type][1]
    while free && dequeue(ready_queue[type], &task_id):
      core = ctz(free)
      slot = (free_bitmap[type][0] & mask) ? 0 : 1
      exe_slots[core*2+slot].task_id = task_id
      asc_threadfence()
      CAS(exe_slots[core*2+slot].state, IDLE, CLAIMED)
      free_bitmap[type][slot] &= ~mask
      free &= ~mask

  if completed_cnt >= total: break
```

### 3.5 AIC/AIV executor 阶段

每个 executor core 轮询自己的 slot：
```
core_id = get_coreid()
exe_type = (core_id < 32) ? CUBE : VECTOR
local_core = core_id - (exe_type == CUBE ? 0 : 32)

while all_done == 0:
  for slot in [0, 1]:
    s = &exe_slots[local_core * 2 + slot]
    if atomicLoad(s.state) == CLAIMED:
      CAS(s.state, CLAIMED, DONE)   // 立即完成
      msg_bitmap[exe_type][slot] |= (1 << local_core)
```

---

## 4. 跨核通信协议

### 4.1 Executor slot CAS 协议

```
dispatch:    CAS(slot.state, IDLE, CLAIMED)    → 成功则 task 已派发
executor:    CAS(slot.state, CLAIMED, DONE)    → 立即完成
dispatch:    atomicLoad(slot.state) == DONE     → 清除, 恢复 IDLE
```

### 4.2 DCCI 规则

- dispatch 写 `exe_slots[core].task_id` 后做 `dcci + dsb`，再 CAS state
- executor CAS state 后读 `task_id`（CAS 是原子的，不需要 DCCI）
- executor 写 `msg_bitmap`（原子 OR，不需要 DCCI）
- dispatch 读 `msg_bitmap`（原子 load）

### 4.3 Queue 协议

cutter 写 `ready_queue`，dispatch 读 `ready_queue`：使用 spinlock（CAS）。
dispatch 写 `completed_queue`，cutter 读 `completed_queue`：同上。

---

## 5. Qwen3 Task Graph (tier 0, 3096 tasks)

### 5.1 Task 序列

| 段 | 算子 | 类型 | 每 tile/row 数 | × 数量 | 小计 |
|----|------|------|----------------|--------|------|
| 1 | RMSNorm | VECTOR | 1 | 6 tiles | 6 |
| 1 | Q-Proj | CUBE | 20 | 6 | 120 |
| 1 | K-Proj | CUBE | 8 | 6 | 48 |
| 1 | V-Proj | CUBE | 8 | 6 | 48 |
| 1 | QK-Norm | VECTOR | 1 | 6 | 6 |
| 2 | ROPE | VECTOR | 1 | 90 rows | 90 |
| 2 | QK-Matmul | CUBE | 4 | 90 | 360 |
| 2 | Softmax | VECTOR | 4 | 90 | 360 |
| 2 | SV-Matmul | CUBE | 4 | 90 | 360 |
| 2 | Online-Softmax | VECTOR | 4 | 90 | 360 |
| 3 | Out-Proj | CUBE | 40 | 6 | 240 |
| 3 | Post-RMSNorm | VECTOR | 1 | 6 | 6 |
| 3 | Gate-Proj | CUBE | 34 | 6 | 204 |
| 3 | Up-Proj | CUBE | 34 | 6 | 204 |
| 3 | SILU | VECTOR | 34 | 6 | 204 |
| 3 | Down-Proj | CUBE | 40 | 6 | 240 |
| 3 | Down-Proj-Res | VECTOR | 40 | 6 | 240 |
| **总计** | | | | | **3096** |

CUBE tasks: 1824, VECTOR tasks: 1272

### 5.2 前驱边（27450 edges）

主要依赖模式：
- RMSNorm → Q/K/V-Proj（每个 Proj 依赖本 tile 的 RMSNorm）
- Q-Proj + K-Proj → QK-Norm
- QK-Norm + V-Proj → ROPE
- ROPE → QK-Matmul → Softmax → SV-Matmul + ROPE → Online-Softmax
- Online-Softmax (所有 row in tile) → Out-Proj
- Out-Proj → Post-RMSNorm → Gate/Up-Proj → SILU → Down-Proj → Down-Proj-Res

---

## 6. 文件结构

```
simt_case/
├── design_v2.md                      # 本文档
├── run.sh
├── common/
│   └── simt_case_protocol.h          # GM 数据结构
├── kernel/
│   └── simt_case_kernel.cpp          # 混合 ELF: AIV VF + AIC Scalar
├── host/
│   └── simt_case_host.cpp            # AICPU alloc + launch + verify
└── ccec/
    └── build.sh
```

---

## 7. 编译方案

### 7.1 CCEC 编译

```bash
# AIC (Cube) — executor entry
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-c310-cube \
    -DSC_WARP_COUNT=8 \
    -I$ASCEND_HOME_PATH/x86_64-linux/include -Icommon \
    -o build/aic.o kernel/simt_case_kernel.cpp

# AIV (Vector + SIMT) — VF + Scalar entry
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-c310-vec \
    -DSC_WARP_COUNT=8 \
    -I$ASCEND_HOME_PATH/x86_64-linux/include -Icommon \
    -o build/aiv.o kernel/simt_case_kernel.cpp

# Link
ld.lld -m aicorelinux -Ttext=0 -static -o build/kernel.o build/aic.o build/aiv.o
```

### 7.2 Host 编译

```bash
g++-15 -O2 -std=c++17 -DSC_WARP_COUNT=8 \
    -Icommon -I$ASCEND_HOME_PATH/include \
    host/simt_case_host.cpp \
    -L$ASCEND_HOME_PATH/x86_64-linux/lib64 -lascendcl -lruntime -ldl \
    -o build/host
```

---

## 8. 运行方案

```bash
# 4 warp
SC_WARP_COUNT=4 ./run.sh build
SC_WARP_COUNT=4 ./run.sh run 0

# 8 warp
SC_WARP_COUNT=8 ./run.sh build
SC_WARP_COUNT=8 ./run.sh run 0

# 通过 task-submit
./run.sh run-task
```

### 预期输出

```
[INIT] Qwen3-14B decode: 3096 tasks, 27450 edges, 1824 cube + 1272 vec
[DEVICE] id=0 soc=Ascend950PR_9579
[ALLOC] state=0x... (2MB)
[LAUNCH] blocks=32 warps=8
[PASS] run=1 tasks=3096 done=3096 kernel_us=XXX
[PERF] passes=1/1 median_us=XXX
```

---

## 9. 验证项

1. `all_done == 1`
2. `completed_cnt == 3096`
3. `report_desc_writes == 3096`（desc phase 完成）
4. `report_cutter_ready == 3096`（cutter 提交所有 task）
5. `report_dispatch_sent == 3096`（dispatch 派发所有 task）
6. `report_executor_done == 3096`（executor 完成所有 task）
7. 所有 `exe_slots[].state == IDLE`（恢复初始状态）
