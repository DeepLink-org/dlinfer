---
name: graph-mode-internals
description: 理解 lmdeploy+dlinfer 中 graph mode 的完整流程，涵盖 runner 架构、buffer 管理、vendor 差异与常见陷阱。
---
# graph-mode-internals

本技能说明 lmdeploy+dlinfer 中 graph mode 的端到端工作原理，涵盖 runner
层、buffer 层、capture/replay 流程以及各 vendor 的具体差异。目标是理解，
而不只是查阅实现细节。

---

## 背景

**什么是 graph mode？**
Graph mode 将一段计算过程捕获为静态图，之后每次执行只需回放（replay），无
需 Python 层的调度开销。在实际推理中，这意味着每个 decode 步骤可以复用预
编译的执行计划，从而降低单步延迟。

**为什么只用于 decode，不用于 prefill？**
Prefill 的序列长度因请求而异，变化范围极大。若要为每种可能的长度单独捕获
一张图，需要大量分桶，占用大量编译时间和显存。Decode 则不同：每个请求每步
只生成一个新 token，即 `q_seqlen = 1`，因此只需按 batch size 分桶，代价可
以接受。

**Eager mode** 完全跳过图捕获，通过 Python dispatch 直接执行算子，是参考
执行路径。

---

## 代码组织

### lmdeploy（基类与 CUDA 实现）

- **`CudaGraphMeta`**（`lmdeploy/pytorch/models/utils/cudagraph.py`）——
  存储图配置的 dataclass：`max_batchs`、`max_tokens`、`num_blocks`、
  `device`、`input_buffers`、`output_buffers`，以及 MLA、SSM、MRoPE
  等可选标志。
- **`CudaGraphMixin`**（同一文件）—— 定义五个方法并提供默认 CUDA 实现的
  mixin 类：
  - `support_cuda_graph` —— 判断当前步骤是否使用 graph mode（默认：
    decode 时返回 True）
  - `make_buffers_cudagraph` —— 分配固定形状的 tensor，供后续所有
    replay 步骤用作图输入
  - `fill_buffers_cudagraph` —— 在 capture 或 replay 前，将真实的
    per-step 数据拷贝到固定 buffer 中
  - `update_context_cudagraph` —— 更新 `StepContext` 中的字段，使其
    指向 buffer tensor
  - `get_outputs_cudagraph` —— replay 结束后，将完整输出 buffer 按
    实际 token 数截取
- **`GraphRunner`**（`lmdeploy/pytorch/backends/graph_runner.py`）——
  基类，`__call__` 直接调用 `self.model(**kwargs)`（无图）
- **`CUDAGraphRunner`**（`lmdeploy/pytorch/backends/cuda/graph_runner.py`）
  —— 完整的 CUDA 实现，包含 `CUDASingleGraphRunner`
  （使用 `torch.cuda.CUDAGraph`）和 batch size 分桶逻辑

### dlinfer（各 vendor 扩展）

所有 vendor 在模块导入时以 monkey-patch 方式替换三个 buffer 方法：

```python
CudaGraphMixin.make_buffers_cudagraph   = Vendor_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph   = Vendor_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = Vendor_update_context_cudagraph
```

Ascend 还额外提供了 **`AscendGraphRunner`**，继承自 `GraphRunner`，内部
使用 `AscendSingleGraphRunner`（基于 `torch.npu.NPUGraph`）。Camb、MACA
和 PPU 则复用 lmdeploy 的 `CUDAGraphRunner`。

各 vendor 选择 runner 类的入口是
`lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py` 中的
**`op_backend.build_graph_runner()`**。

---

## Runner 层

### Batch size 分桶（`compatible_size`）

图捕获以 batch size 为键。为最大化图复用，实际 batch size 在查找或创建图
之前会向上取整到最近的桶：

- **Ascend**（`AscendGraphRunner.get_ascend_compatible_size`）：三段策略
  —— ≤ 16 时取 2 的幂次，≤ 256 时按 16 对齐，> 256 时按 256 对齐
- **Camb / MACA / PPU**（通过 `CUDAGraphRunner`）：纯粹的 2 的幂次

### `_runner_map` 与图的生命周期

`_runner_map` 以 `(compatible_batch_size, is_decoding, ...)` 为键，映射
到单个图 runner。首次遇到时捕获图；后续遇到时直接回放已缓存的图。

---

## Buffer 层

### Tensor 的两类

| 类别 | 形状随 batch size 变化？ | 需要 buffer？ |
|---|---|---|
| KV cache（`past_key_values`） | 否——启动时按最大容量分配 | 否 |
| `q_seqlens`、`q_start_loc`、`block_offsets`、`kv_start_indices`、`input_ids` 等 | 是 | 是 |

KV cache 直接透传，无需 buffer。形状随 batch size 变化的 tensor 必须由
固定形状的 buffer 托底，以确保捕获的图始终看到相同的内存地址和形状。

### 三个 buffer 方法

**`make_buffers_cudagraph`** —— 在图捕获准备阶段调用一次。在设备上分配
固定形状的 tensor（按 `max_batchs` / `max_tokens` 大小），并存入
`graph_meta.input_buffers`。

**`fill_buffers_cudagraph`** —— 在每次 capture 和每次 replay 前调用。将
真实数据从 forward 输入拷贝到预分配的 buffer 中。对填充槽位使用安全默认值
（例如，padding seqlen 填为 `max_tokens // max_batchs`；`kv_start_indices`
初始化为 -1，防止 padding 槽位污染 KV cache 的 slot 0）。

**`update_context_cudagraph`** —— 在每次 capture 和每次 replay 前调用。
更新 `StepContext`，使其指向 buffer tensor，以便下游算子（如 attention）
读取正确的内存。

如果引入新的随 batch size 变化的 tensor，三个方法都需要同步更新。

---

## Capture 流程

```text
GraphRunner.__call__
  └─ compatible_size = get_compatible_size(batch_size)
       └─ _runner_map[compatible_size] 不存在 → 创建 AscendSingleGraphRunner
            （Camb / MACA / PPU 使用 CUDASingleGraphRunner）
            │
            ├─ make_buffers_cudagraph(graph_meta)  ← 一次性分配固定 buffer
            │
            ├─ fill_buffers_cudagraph(...)          ← 将真实数据写入 buffer
            │
            ├─ update_context_cudagraph(...)        ← StepContext 指向 buffer
            │
            ├─ warmup forward（图范围之外）
            │
            └─ with torch.cuda.graph() / torch.npu.NPUGraph():
                 model.forward(...)                 ← 算子在此处被捕获
                 make_output_buffers(output)        ← 保存输出 tensor 引用
```

---

## Replay 流程

```text
GraphRunner.__call__
  └─ compatible_size = get_compatible_size(batch_size)
       └─ _runner_map[compatible_size] 存在 → AscendSingleGraphRunner.forward()
            │
            ├─ fill_buffers_cudagraph(...)     ← 更新 buffer 内容
            │
            ├─ update_context_cudagraph(...)   ← 重新指向 StepContext
            │
            ├─ [仅 Ascend] 原地更新 kv_seqlens（见下节）
            │
            ├─ _graph.replay()                 ← 执行捕获的算子序列
            │
            └─ get_outputs_cudagraph(...)      ← 按实际 token 数截取输出
```

> **说明**：`get_outputs_cudagraph` 是一个简单的输出截取步骤。它读取
> `output_buffers['hidden_states']` 并截取 `[:, :num_tokens]`。对于大多数
> vendor，与 lmdeploy 默认实现相同。

---

## Ascend —— Replay 期间的 kv_seqlens 更新

对于 Camb 和 MACA，在 replay 前将新值写入设备 buffer 即可——图
replay 时会自动读取 buffer 中的最新值。Ascend 则不同：attention 算子的
`actual_seq_lengths_kv` 是 CPU tensor 或 list，而非 NPU buffer 的一部分，
写 NPU buffer 无法触达这个 CPU 侧参数，因此必须通过专门的 update API 将新
值显式推入已捕获的图中。

通过 `aclgraph_use_torch_npu_update()` 在运行时选择以下两种机制之一：

**torch_npu < 2.8.0.post1** —— 使用底层 ACL graph task update API：

```python
graph_task_update_begin(graph_handle)
update_attn_params(kv_seqlens, ...)  # 通过 ACL 写入
graph_task_update_end(graph_handle)
```

**torch_npu ≥ 2.8.0.post1** —— 使用更高级的 torch_npu graph update API：

```python
graph.update(cpu_update_input=[{"actual_seq_lengths_kv": kv_seqlens}])
```

---

## 各 Vendor 对比

| 项目 | Ascend | Camb | MACA |
|---|---|---|---|
| Runner 类 | `AscendGraphRunner`（自定义） | `CUDAGraphRunner`（lmdeploy） | `CUDAGraphRunner`（lmdeploy） |
| Graph API | `torch.npu.NPUGraph` | `torch.cuda.CUDAGraph` | `torch.cuda.CUDAGraph` |
| `compatible_size` 策略 | 三段式（2幂/16对齐/256对齐） | 纯 2 的幂次 | 纯 2 的幂次 |
| `fill_buffers` 中 `attn_metadata` 截取 | 不截取（使用完整 buffer） | 截取到 `[:new_batch_size]` | 截取到 `[:new_batch_size]` |
| `kv_start_indices` 形状 | `(max_batches,)` | `(max_batches,)` | `(max_batches, 1)` |
| `fill_buffers` 中 `max_kv_seq_len` | 保持原值 | 设为 -1 | 保持原值 |
| `x_active_mask` buffer | 有 | 无 | 无 |
| Replay 期间 `kv_seqlens` 更新方式 | `update_attn_params` 或 `graph.update()`（依 torch_npu 版本） | 写入 buffer | 写入 buffer |

---

## 注意事项

1. **`kv_start_indices` 必须初始化为 -1，而非 0。** Index 0 是合法的
   KV cache slot；若 padding 槽位初始化为 0，会悄无声息地污染它。

2. **Camb 的 `max_kv_seq_len` 必须设为 -1。** 此整数在捕获时会作为
   常量节点固化到图中。`torch_mlu_ops` API 约定：值 ≤ 0 表示"从
   `kv_seqlens` tensor 动态计算最大值"；若 capture 时填入真实最大值，
   replay 时每一步都会使用该固化的错误常量。

3. **三个 buffer 方法必须同步更新。** 如果引入新的随 batch size 变化的
   tensor，`make_buffers` 需要分配 buffer，`fill_buffers` 需要写入数据，
   `update_context` 需要让 `StepContext` 指向它。任意一步缺失都会导致错
   误行为或悄悄读取到旧数据。

4. **图以 `compatible_size` 为键，而非实际 batch size。** Batch size 向
   上取整到桶。不要将 `new_batch_size` 直接与 `max_batchs` 比较，应使用
   compatible-size 逻辑。

5. **Ascend kv_seqlens 更新的版本检查。** 调试 Ascend graph mode 中
   attention 输出错误的问题时，先确认 torch_npu 版本，再验证
   `AscendSingleGraphRunner` 走的是哪条更新路径。

6. **Eager mode 始终可作为参考。** 若 graph mode 产生错误输出，以
   `eager_mode=True` 运行同一步骤，确认 bug 是在图捕获/回放中，还是在
   底层算子中。
