# SIMT Qwen3 Decode Case — 可运行文件清单

> 生成时间：2026-08-12
> 验证状态：A5 上板 3/3 PASS（4/8/16 warps 均通过）

---

## 1. 目录结构

```
simt_case/
├── run.sh                         # 统一入口：build / run / run-task
├── ccec/
│   └── build.sh                   # CCEC + GCC-15 编译脚本
├── common/                        # 共享头文件（设备 + 主机）
│   ├── g0_full_pa.h               # FullPaState 结构定义 + BuiltState/ClaimedState/DoneState
│   ├── full_pa_model.h            # TaskKind enum + TaskKindAt + TaskShape + dispatch IDs
│   ├── full_pa_exec_protocol.h    # ExecutionToken + SharedExecCell + AtomicLine + EncodeExecState
│   ├── shared_protocol.h          # 基础协议 (EMPTY→BUILDING→BUILT→CLAIMED→DONE 状态机)
│   ├── simt_case_protocol.h       # 自有 GM 结构 (ScState — 未使用，保留)
│   └── qwen3_task_model.h         # Qwen3 task graph builder (统计用，未接入执行)
├── kernel/
│   ├── g0_full_pa_kernel.cpp      # 混合 ELF kernel 源码 (3621 行，17KB entry)
│   ├── full_pa_workloads.h        # PTO workload 函数 (RunG0CubeMatmul / RunG0VectorAdd)
│   └── simt_case_kernel.cpp       # 自有 kernel (未使用，保留)
├── host/
│   └── simt_case_host.cpp         # ACL host 源码 (3850 行，InitializeState + AclSession + main)
├── design.md                      # 原始设计文档
├── design_v2.md                   # 简化设计文档
└── build/                         # 编译产物 (gitignore)
    └── ccec/
        ├── simt_case_aic.o        # AIC (Cube) object
        ├── simt_case_aiv.o        # AIV (Vector+SIMT) object
        ├── simt_case_kernel.o     # 链接后的 mixed ELF binary
        └── simt_case_host         # GCC-15 编译的 host 可执行文件
```

---

## 2. 文件依赖关系

### 编译时依赖

```
g0_full_pa_kernel.cpp
  ├── <pto/common/kernel_meta.hpp>     # CANN PTO 框架
  ├── <pto/pto-inst.hpp>               # CANN PTO 指令 (TLOAD/TMATMUL/TSTORE)
  ├── <cce_aicore_intrinsics.h>        # AICore 内置函数 (atomicAdd/CAS/dcci/dsb)
  ├── "simt_api/asc_simt.h"            # SIMT VF API (仅 __DAV_VEC__)
  ├── "../common/g0_full_pa.h"         # FullPaState + 状态编码
  │     └── "full_pa_model.h"          # TaskKind + TaskShape + dispatch
  │           └── "full_pa_exec_protocol.h"  # ExecutionToken + SharedExecCell
  │                 └── "shared_protocol.h"  # 基础协议头
  └── "full_pa_workloads.h"            # RunG0CubeMatmul / RunG0VectorAdd

simt_case_host.cpp
  ├── <acl/acl.h> + <acl/acl_rt.h>     # ACL runtime
  ├── "../common/g0_full_pa.h"         # FullPaState (同上)
  └── "../common/qwen3_task_model.h"   # Qwen3 graph (仅统计)
```

### 运行时依赖

```
simt_case_host (可执行文件)
  ├── libascendcl.so                   # ACL 运行时库
  ├── libruntime.so                    # runtime 库
  └── simt_case_kernel.o (ELF binary)  # AICore kernel binary

simt_case_kernel.o (mixed ELF)
  ├── simt_case_aic.o                  # AIC entry: simt_cross_core_g0_0_mix_aic
  └── simt_case_aiv.o                  # AIV entry: simt_cross_core_g0_0_mix_aiv
```

---

## 3. 编译命令

### 环境要求

| 项目 | 要求 | 当前值 |
|------|------|--------|
| ASCEND_HOME_PATH | CANN 工具包路径 | `/usr/local/Ascend/cann-9.2.0` |
| ccec | CCEC 编译器 | `$ASCEND_HOME_PATH/bin/ccec` |
| ld.lld | AICore 链接器 | `$ASCEND_HOME_PATH/bin/ld.lld` |
| g++-15 | Host 编译器 | `/usr/bin/g++-15` |
| 设备 | A5 / Ascend950 | `npu-smi info` 确认 |
| task-submit | 受限环境必需 | `/usr/local/bin/task-submit` |
| 架构缓存 | 预检跳过 | `a5|Ascend950PR_9579|Ascend950` |

### 一键编译

```bash
cd /data/pyptouser/xionglejin/project/esl_proxy/simt_case

# 默认 16 warps
./run.sh build

# 或指定 warp 数
SC_WARP_COUNT=4  ./run.sh build   # 4 warps
SC_WARP_COUNT=8  ./run.sh build   # 8 warps
SC_WARP_COUNT=16 ./run.sh build   # 16 warps (默认)
```

### 手动编译（等价于 build.sh）

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"
WARP_COUNT=16

# 1. 编译 AIC (Cube)
"$CCEC" -c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only --cce-aicore-arch=dav-c310-cube \
    -DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$WARP_COUNT \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -mllvm -cce-aicore-dcci-before-kernel-end=false \
    -I"$ASCEND_HOME_PATH/x86_64-linux/include" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc/include" \
    -Icommon -Ikernel \
    -o build/ccec/simt_case_aic.o \
    kernel/g0_full_pa_kernel.cpp

# 2. 编译 AIV (Vector + SIMT)
"$CCEC" -c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only --cce-aicore-arch=dav-c310-vec \
    -DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$WARP_COUNT \
    [同上 mllvm 和 -I 参数] \
    -o build/ccec/simt_case_aiv.o \
    kernel/g0_full_pa_kernel.cpp

# 3. 链接 mixed ELF
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o build/ccec/simt_case_kernel.o \
    build/ccec/simt_case_aic.o \
    build/ccec/simt_case_aiv.o

# 4. 编译 host
g++-15 -O2 -std=c++17 -Wall -Wextra -Wno-deprecated-declarations \
    -DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$WARP_COUNT \
    -Icommon \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    host/simt_case_host.cpp \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl \
    -o build/ccec/simt_case_host
```

---

## 4. 运行命令

### 直接运行（有设备访问权限）

```bash
cd /data/pyptouser/xionglejin/project/esl_proxy/simt_case

# 默认 256 batches, 1 builder, 1 run
./run.sh run 0

# 多次运行
SC_RUNS=3 ./run.sh run 0

# 自定义参数
SC_BATCHES=256 SC_BUILDERS=1 SC_RUNS=10 ./run.sh run 0
```

### 通过 task-submit 运行（受限环境）

```bash
cd /data/pyptouser/xionglejin/project/esl_proxy/simt_case
./run.sh run-task

# 或手动
task-submit --device 0 --timeout 600 --run \
    "cd /data/pyptouser/xionglejin/project/esl_proxy/simt_case && SC_RUNS=3 ./run.sh run \$TASK_DEVICE"
```

### 预期输出

```
[QWEN3] decode graph (tier 0): 3102 tasks (1830 cube + 1272 vec), 6 tiles, 90 rows
[DEVICE] id=0 soc=Ascend950PR_9579 topology=32*(1AIC+2AIV) builders=1
         simt_threads_per_builder=512 warps_per_builder=16
         state_bytes=32007296 workspace_bytes=12713984
         batches=256 runs=3
[PASS] run=1 active_tasks=1280 kernel_tasks=1024 kernel_event_us=2645
[PASS] run=2 active_tasks=1280 kernel_tasks=1024 kernel_event_us=2628
[PASS] run=3 active_tasks=1280 kernel_tasks=1024 kernel_event_us=2760
[PERF] median_us=2645.7 min_us=2628.5 max_us=2760.2
[SUMMARY] passes=3/3 same_address_reuse=validated
```

---

## 5. 环境设置

```bash
# CANN 路径
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0

# 架构预检缓存（跳过 onboarding）
echo 'a5|Ascend950PR_9579|Ascend950' > /tmp/onboard-arch-precheck.cache

# 本地工具（如果需要 llvm-bcanalyzer）
export PATH="/tmp/opencode/localtools/bin:$PATH"

# 验证
which ccec          # /usr/local/Ascend/cann-9.2.0/bin/ccec
which ld.lld        # /usr/local/Ascend/cann-9.2.0/bin/ld.lld
which g++-15        # /usr/bin/g++-15
which task-submit   # /usr/local/bin/task-submit
npu-smi info        # 确认 A5 设备可用
```

---

## 6. 性能数据

| Warp Count | Threads/Builder | Median (us) | Min (us) | Max (us) | Pass Rate |
|------------|-----------------|-------------|----------|----------|-----------|
| 4 | 128 | 4286 | 4280 | 4570 | 3/3 |
| 8 | 256 | 3009 | 2990 | 3230 | 3/3 |
| 16 | 512 | 2646 | 2628 | 2760 | 3/3 |

配置：256 batches, 1 builder, 1024 kernel tasks, 32 AIC + 64 AIV blocks

---

## 7. 执行流程

```
AICPU Host:
  1. aclrtMalloc FullPaState (32MB) + workspace (12MB)
  2. InitializeState: 0xA5 poison + guards + 1280 tasks + 1024 dispatch IDs
  3. aclrtMemcpy H2D (state + workspace)
  4. aclrtLaunchKernelWithHostArgs(32 blocks)

AIV0 (block0 subblock0) — SIMT VF:
  16 warps × 32 threads = 512 threads
  - Warp-interleaved task construction (TLOAD/TSTORE + CAS BuiltState publish)
  - Strict insert chain (task_id order across warps)
  - set_flag(PIPE_V, PIPE_S) → Scalar waits

AIV0 Scalar (after VF):
  - RunExecutor: token-based executor loop (4 tokens/owner)
    - AdvanceToken: WaitingBuilt → Binding → WaitingFanin → EngineInflight → Completing
    - RunClaimedWorkload: StoreDev64(poison) → RunG0CubeMatmul/VectorAdd → LoadDev64(checksum)
    - PublishExecutionWitness + completion flag
  - ArriveAndDrain: 16-group fan-in + root_finished

AIC0-31 (block0-31 AIC):
  - RunExecutor (same as AIV but Cube engine)
  - RunG0CubeMatmul workload

AIV2-63 (block1-31 subblock1):
  - RunExecutor (Vector engine)
  - RunG0VectorAdd / RunG0VectorMultiply workload

Host (after kernel):
  5. aclrtSynchronizeStream
  6. aclrtMemcpy D2H (state + workspace)
  7. Validate: root_finished + done_count + workload checksums + role results
```

---

## 8. 关键编译产物验证

```bash
# 验证 mixed ELF 有 2 个 GLOBAL FUNC
readelf --symbols --wide build/ccec/simt_case_kernel.o | awk '$5=="GLOBAL" && $4=="FUNC"'
# 期望: simt_cross_core_g0_0_mix_aic + simt_cross_core_g0_0_mix_aiv

# 验证有 .rodata (tile descriptors)
readelf --sections --wide build/ccec/simt_case_kernel.o | grep rodata
# 期望: .rodata (304B)

# 验证 AIV 有 TLV 0x10 (workspace) + TLV 0x0c=4 (MIX_VF)
readelf -x ".ascend.meta.simt_cross_core_g0_0_mix_aiv" build/ccec/simt_case_kernel.o
# 期望: 0c000400 04000000 (MIX_VF=4) + 10000c00 (workspace TLV)

# 验证 KernelArgSize = 8 bytes (1 pointer)
readelf -x "__CCE_KernelArgSize" build/ccec/simt_case_kernel.o
# 期望: 08000000 08000000
```
