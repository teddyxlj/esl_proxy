# 架构重新设计 — AICPU (ARM CPU) + SIMT VF + AICore executor

## 理解

AICPU = A5 device 上的 ARM CPU（独立处理器），不是 host x86，不是 AIC Scalar。
通过 CANN 的 `aclrtRegisterCpuFunc` / `rtsLaunchCpuKernel` 机制加载 .so 运行。

## 架构

```
Host x86:
  1. 编译 AICPU .so (ARM cross-compile)
  2. 编译 AICore mixed ELF (CCEC)
  3. aclrtMalloc GM
  4. 加载 AICPU .so + 加载 mixed ELF
  5. 启动 AICPU kernel (alloc + cutter + dispatch)
  6. 启动 AICore kernel (SIMT VF desc + executor)
  7. 等待完成, D2H verify

AICPU (ARM CPU on device):
  alloc:  初始化 GM (g_basic_buf + g_predecessors + g_task_tensor_buf)
  cutter: 消费 g_predecessors → 建 successor 链 → ready_queue
  dispatch: ready_queue → executor slots (CAS IDLE→CLAIMED)
            msg_bitmap → completed_queue → resolve_dep

AIV0 (AICore Vector, SIMT VF):
  desc: warp-interleaved 写 g_basic_buf + g_predecessors (qwen3_14b_decoder_desc_ap.h)

AIC0-31 + AIV2-63 (AICore executor):
  poll exe_slots → CAS CLAIMED→DONE (immediate complete)
```

## 问题

AICPU .so 和 AICore mixed ELF 是两套独立的编译和加载机制。
AICPU 代码用 ARM 交叉编译器编译为 .so。
AICore 代码用 CCEC 编译为 mixed ELF。
两者通过 GM (Global Memory) 共享数据。

## 需要确认

1. AICPU .so 的编译工具链是什么？(aarch64-linux-gnu-gcc?)
2. AICPU .so 的加载方式？(JSON config + rtsBinaryLoadFromFile?)
3. AICPU 和 AICore kernel 能否同时运行？(并发 launch?)
4. GM 地址如何在两者之间共享？(aclrtMalloc 返回的地址?)
```

是否确认这个方向？如果是，我需要：
1. 找到 ARM 交叉编译器
2. 将 cutter.c + dispatch.c + alloc 逻辑编译为 AICPU .so
3. 将 desc + executor 编译为 mixed ELF
4. Host 同时加载和启动两者
