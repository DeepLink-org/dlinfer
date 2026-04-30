---
name: support-new-model-zh
description: 在国产 AI 硬件（Ascend / CAMB / MACA）上通过 dlinfer 适配一个新模型（该模型已在 lmdeploy CUDA backend 上支持）。
---

你正在帮助用户将一个新的 LLM 或 VLM 适配到 dlinfer 支持的硬件后端（Ascend NPU、CAMB MLU、MACA GPU）。该模型已通过 lmdeploy 在 CUDA 上运行——你的任务是找出目标 vendor 缺少什么，并补全它。

---

## 第一步 — 收集信息

询问用户：
1. **要适配哪个模型**？（提供模型在 `lmdeploy/pytorch/models/` 中的名称，例如 `qwen3`、`deepseek_v2`）
2. **目标 vendor 是哪些**？（ascend / camb / maca，可多选）

两个问题都得到回答后再继续。

---

## 第二步 — 分析模型

使用 Read/Bash 工具自行读取以下所有文件，不要让用户去读：

```
lmdeploy/lmdeploy/pytorch/models/<model>.py
lmdeploy/lmdeploy/pytorch/backends/dlinfer/op_backend.py
lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py   ← 每个目标 vendor 各读一份
```

完整调用链为：`models/<model>.py` → `lmdeploy/pytorch/nn/` → `backends/dlinfer/`
→ `kernels/dlinfer/` → `dlinfer/ops/` → `vendor/`。如果某个 model 层和 backend op 之间的对应关系不清楚，顺着 `lmdeploy/pytorch/nn/` 中间层追踪。`lmdeploy/pytorch/kernels/default/` 包含 CUDA 参考实现，在编写新 vendor op 时可用作规格参考，按需读取。

### 从 `models/<model>.py` 中识别：

- 模型使用的所有非平凡算子：attention 变体（paged、flash、MLA）、MLP 激活函数、RMS norm 变体、MoE routing、rotary embedding 变体（标准、MROPE、多尺度）、量化 op 等。
- 模型是否通过 `StepContext` 或 `attn_metadata` 传递了超出标准字段的新输入。标准字段为：`input_ids`、`position_ids`、`block_offsets`、`q_seqlens`、`kv_seqlens`、`kv_start_indices`。已知的扩展字段：`state_ids`（SSM 模型）、`mrope_position_ids`（MROPE 模型）、`cu_seqlens` / `has_initial_state`（Gated Delta Network）。

### 从通用 `op_backend.py` 中检查：

- `get_layer_impl_builder()`：哪些 `OpType` 已有 dlinfer `Impl`。与上面的 op 列表对比，找出缺口 → **路径 A**。

### 从 `<vendor>/op_backend.py` 中逐项检查：

- **`update_step_context()`**：该方法负责在每次推理步骤中构建 `attn_metadata`（Ascend 上还包括 `moe_metadata`）。需要仔细确认它是否正确处理了新模型所需的所有字段。若模型引入了新的 context 字段或新的 attention 模式（例如类似 `is_gated_delta` 的标志），则需要扩展此方法 → **路径 B**。
- **`get_k_block_shape()` / `get_v_block_shape()`**：确认 KV cache 的内存布局与模型 attention 实现的期望一致。不同 vendor 甚至同一 vendor 的不同 SoC 版本（Ascend A2 vs A3、310P）可能使用不同的 layout → **路径 B**（如不匹配）。
- **`AscendKVQuantMeta`**（仅 Ascend）：若模型使用 KV cache 量化，且 scale/offset 格式与当前实现不同 → **路径 B**。

在动手写代码之前，先向用户汇报分析结果：
- Op 缺口（→ 路径 A）
- Vendor `op_backend.py` 缺口（→ 路径 B）
- Framework 层面缺口（→ 路径 C）

---

## 路径 A — 补充缺失的 op（4 层栈）

对每个在 `get_layer_impl_builder()` 中缺失的 op，按从上到下的顺序逐层实现。

### 第一层 — `lmdeploy/lmdeploy/pytorch/backends/dlinfer/`

新增 `XxxImpl`（继承 lmdeploy 基类 `Impl`）和 `XxxBuilder`（包含 `build()` 方法）。
在 `op_backend.py` 的 `get_layer_impl_builder()` dispatcher 中注册该 Builder。
参考：`activation.py`（最简单）、`norm.py`、`attention.py`（最复杂）。

### 第二层 — `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/`

新增一个薄 wrapper 函数，调用 `dlinfer.ops.<op_name>(...)`，并在 `__init__.py` 中导出。

### 第三层 — `dlinfer/dlinfer/ops/llm.py`

用 `@register_custom_op("dlinfer::<op_name>", [...])` 注册新 op，函数体转发到 `vendor_ops_registry["<op_name>"]`。
**此处的字符串 key 必须与第四层中被装饰函数的名称完全一致。**

### 第四层 — `dlinfer/dlinfer/vendor/<vendor>/`

添加带 `@register_ops(vendor_ops_registry)` 装饰的实现，调用 vendor 的 native op：
- **Ascend**：`torch.ops.npu.*`，参考 `vendor/ascend/torch_npu_ops.py`
- **CAMB**：`tmo.*`（`torch_mlu_ops`），参考 `vendor/camb/camb_ops.py`
- **MACA**：`mcoplib.*`，参考 `vendor/maca/maca_ops.py`

**Ascend**：在 `torch_npu_ops.py` 中新增任何算子之前，先请用户提供该算子的官方 NPU 文档。严格按照文档实现：参数名称、tensor shape 约束、dtype 约束无法从现有代码中推断，写错会引发难以定位的运行时错误。

逻辑较复杂时（如 Ascend 带 graph mode 记录的 attention），拆分到辅助模块（如 `vendor/ascend/attention.py`）并在 `torch_npu_ops.py` 中导入。

---

## 路径 B — Vendor-specific `op_backend.py` 修改

文件：`lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py`

以下三个子方向相互独立，分别评估。

### B1 — `update_step_context()`：新 context 字段或 attention 模式

当新模型需要 `attn_metadata` 中有当前实现未填充的字段时，扩展 `update_step_context()`：
- 在方法内部按已有的 helper 函数模式计算新字段。
- 在方法末尾构造 `attn_metadata` 时将新字段传入。
- Ascend 上若模型引入了新的 MoE 通信模式或并行拓扑，还需扩展 `moe_metadata`。

参考：`is_gated_delta` 分支（添加 `cu_seqlens` 和 `has_initial_state`）、`kv_quant_policy == 8` 分支（填充 `AscendKVQuantMeta`）。

### B2 — `get_k_block_shape()` / `get_v_block_shape()`：KV cache layout

硬件目标确定后这里基本不再改动。跳过，除非新模型引入了现有任何 vendor backend 都无法覆盖的全新 attention 内存布局需求。

### B3 — `AscendKVQuantMeta`：KV 量化（仅 Ascend）

遗留功能，当前正确性未经主动验证。常规新模型适配跳过此项——仅在明确需要 KV cache 量化且已确认该功能可用时再处理。

---

## 路径 C — Framework patch（`dlinfer/dlinfer/framework/lmdeploy_ext/`）

以下三个子模块相互独立，分别评估。

### C1 — cudagraph / aclgraph 缓冲区管理

**触发条件**：仅当模型引入了新的 `StepContext` 字段，且该字段的 **shape 随 batch size 或 seq_len 在运行时动态变化**时才需要处理。shape 固定的 tensor 不需要特殊缓冲区管理。示例：`x_active_mask`（shape `[batch_size]`）是为 Expert Parallelism 支持而添加的——它的尺寸随每步变化，因此需要预分配最大尺寸的 buffer。

- **Ascend**：`framework/lmdeploy_ext/cudagraph/ascend_cudagraph.py`
  - `make_buffers_cudagraph`：以最大尺寸（`max_batches` / `max_tokens`）预分配新字段的 tensor。用运行时尺寸会导致 graph replay 时 shape 不匹配。
  - `fill_buffers_cudagraph`：将运行时数据拷贝到预分配的 buffer 中。
  - `update_context_cudagraph`：将 buffer 写回 step context。
  - 参考：`is_ssm`（`state_ids`）和 `use_mrope`（`mrope_position_ids`）的处理模式。
- **其他 vendor**：在对应的 `camb_cudagraph.py` / `maca_cudagraph.py` 中应用相同模式。

若模型只使用已有的标准字段，跳过此节。

### C2 — Device-specific patch

**触发条件**：模型需要对 lmdeploy 的某个行为做 vendor 级别的覆盖时——例如 Ascend 上不同的 MoE 通信策略、CAMB 上不支持的 sampling op，或硬件特定的 KV cache 格式（如 Ascend 310P NZ 格式）。

- **Ascend**：`framework/lmdeploy_ext/device/ascend.py`
- **CAMB**：`framework/lmdeploy_ext/device/camb.py`

直接在 lmdeploy 类上 patch 对应方法。确保 patch 文件在 `framework/lmdeploy_ext/device/__init__.py` 中被导入。

### C3 — 量化 patch

**触发条件**：仅当模型使用 AWQ 且权重打包格式或 scale layout 与当前 Ascend 实现不兼容时。

文件：`framework/lmdeploy_ext/quants/ascend_awq.py`

该文件 patch 了 `WeightOnlyQLinear`、`MergedAwqLinear`、`AwqLinear`、`QKVAwqLinear`。仅当新模型的量化 checkpoint 使用了当前 patch 无法处理的 layout 时才修改。

---

## 验收 checklist

**路径 A（新 op）：**
- [ ] 每个缺失 op 的 4 层均已实现
- [ ] 通用 `op_backend.py` 的 `get_layer_impl_builder()` 已更新
- [ ] `ops/llm.py` 中的 `vendor_ops_registry` key 与 vendor 文件中被装饰函数名完全一致
- [ ] 新 kernel 已在 `kernels/dlinfer/__init__.py` 中导出

**路径 B（vendor `op_backend.py`）：**
- [ ] `update_step_context()` 已正确填充新模型所需的所有 `attn_metadata` 字段

**路径 C1（graph 缓冲区）：**
- [ ] 新字段已在 `make_buffers_cudagraph` 中以最大尺寸预分配
- [ ] 新字段已在 `fill_buffers_cudagraph` 中填充
- [ ] 新字段已在 `update_context_cudagraph` 中写回 context

**路径 C2（device patch）：**
- [ ] patch 已直接应用到 lmdeploy 类上
- [ ] patch 文件已在 `device/__init__.py` 中导入

**路径 C3（量化 patch）：**
- [ ] 已对照 checkpoint 格式核实权重打包 / scale layout
- [ ] 相关类方法已在 `ascend_awq.py` 中 patch

**通用：**
- [ ] eager mode：模型可正常推理
- [ ] graph mode：模型可正常推理（如该 vendor 支持）
