---
name: precision-align-zh
description: 诊断并修复 lmdeploy+dlinfer 在国产 AI 硬件（Ascend / CAMB / MACA）上的精度问题，通过与参考实现对比找到偏差根因。
---

你正在帮助用户修复 lmdeploy+dlinfer 在国产 AI 硬件后端（Ascend、CAMB 或 MACA）上的精度问题。
参考实现通常是 vllm+vllm-ascend（针对 Ascend）或其他约定的参考框架。目标是找到
lmdeploy+dlinfer 与参考实现的偏差根因并修复。

本 skill 中的示例以 Ascend 为具体硬件。CAMB 和 MACA 适用相同方法论，替换对应的 vendor
路径和算子调用即可。

---

## 第一步 — 收集信息

询问用户：

1. **对齐的是哪个模型**？（例如 `qwen3`、`deepseek_v2`）
2. **目标硬件是哪个**？（ascend / camb / maca）
3. **现象是什么**？——例如从第一个 token 就不一致、几个 token 之后开始乱说、精度评测分数下降了多少分。
4. **并行配置**：使用的是什么 TP / DP / EP？
5. **是否有初步观察**？第一个生成的 token 就已经错误（→ prefill 问题），还是前几个 token 正确之后才出现偏差（→ decode / KV cache 问题）？
6. **单 batch 还是多 batch**？用单个请求（batch_size=1）能否复现问题，还是只有多个请求同时处理时才出现？

以上问题都得到回答后再继续。

---

## 第二步 — 确认环境配置

开始排查之前，先确认对比环境是受控的。两侧除被测框架不同外，其余条件必须完全一致：

| 条件                          | lmdeploy+dlinfer           | vllm+vllm-ascend           |
|-------------------------------|----------------------------|----------------------------|
| 相同的 SoC 版本               | ✓                          | ✓                          |
| 关闭 warmup                   | ✓                          | ✓                          |
| Eager mode                    | ✓（`--eager-mode true`）   | ✓（`--enforce-eager`）     |
| 相同的 TP / DP / EP           | ✓                          | ✓                          |
| `temperature=0`、`top_k=1`    | ✓                          | ✓                          |
| 相同的 prompt / 输入          | ✓                          | ✓                          |

如果任何条件未满足，先修复。Warmup 会在 KV cache 中遗留脏数据；temperature > 0 引入采样随机性——两者都会掩盖真实的精度 bug。

---

## 第三步 — 快速输出对比

用**相同的 prompt** 运行两个框架，直接对比生成的 token 序列。

- **token 一致** → 输出对齐，精度大概率没问题。建议用 opencompass 或 evascope 跑评测分数确认。
- **token 不一致** → 进入第四步。

---

## 第四步 — 诊断根因

根据现象确定排查方向：

| 现象                                               | 最可能的原因                        | 路径   |
|----------------------------------------------------|-------------------------------------|--------|
| 第一个生成的 token 就已经错误                      | Prefill 算子精度问题                | B      |
| 第一个 token 正确，之后开始偏差                    | KV cache 污染或 decode 算子         | A 或 B |
| 偏差随序列长度增加而累积                           | KV cache 污染**或**算子精度问题     | A 或 B |
| 在某个固定深度立即出现偏差                         | 算子精度问题                        | B      |
| 仅在 TP > 1 或 dp×tp / ep 配置下出错              | 通信 / 并行策略 patch               | C      |
| 多 batch 时出错，单 batch 正常                     | Batching / seqlen / masking 问题    | A 或 B |

**重要提示**：偏差随序列长度累积**并不代表**一定是 KV cache 污染。算子精度问题同样会在 decode 步骤中逐渐累积——例如 rope embedding 的 cos/sin 在 CPU 上计算，会让每个 token 的位置编码都略有偏差，最终累积成可见的精度下降（曾在 Qwen30B-A3B 上排查出此问题，LiveCodeBench 评分低 2 分）。不要在排除路径 B 之前就断定是 cache 污染。

**不确定时**：
- 先用单 batch 请求复现，排除 batching 的干扰。
- 从能装下模型权重的最简并行配置开始（具体见路径 C 的并行层次说明）。部分大模型无法在 TP=1 下运行，"最简"指的是能加载权重的最少并行维度。
- 然后从路径 B 的 layer 0 开始排查。

---

## 路径 A — KV cache 污染

KV cache 污染意味着 `fill_kv_cache` 的索引传错了，把某些 token 写到了错误的 cache slot（踩踏）。`fill_kv_cache` 的内部逻辑一般不会出问题——问题几乎总是在传给它的索引上。

### 需要读的文件

- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/fill_kv_cache.py`
- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/pagedattention.py`

重点参数：
- `fill_kv_cache` 的 `kv_start_indices`：每个 token 在 cache 中的 flat slot 索引。
- `prefill_attention` 和 `paged_token_attention` / `paged_attention_fwd` 的 `block_offsets`、`q_start_loc`、`q_seq_len`、`kv_seq_len`。

**多 batch 注意**：如果问题只在多请求时出现，重点核查每个请求的 seqlen 追踪（`q_seq_len`、`kv_seq_len`）和 `kv_start_indices`。per-request 长度错误会导致 attention 从 KV cache 的错误位置读取数据。

### 排查方法

**不要 dump KV cache tensor 本身**——太大了。在可疑层的 `fill_kv_cache` 调用**之前**，dump 以下三个 tensor：

```python
dump("key_states",       key_states)        # shape: [num_tokens, num_kv_heads, head_size]
dump("value_states",     value_states)      # shape: [num_tokens, num_kv_heads, head_size]
dump("kv_start_indices", kv_start_indices)  # shape: [num_tokens]
```

**关键检查**：`kv_start_indices.shape[0]` 必须等于 `key_states.shape[0]`。如果长度不一致，说明索引数量与待写入的 token 数量不匹配，fill 时会发生踩踏，导致后续 decode 步骤的 cache 内容被污染。

---

## 路径 B — 算子精度

目标是找到 lmdeploy+dlinfer 与参考实现**第一次出现差异的算子**。

**先用单 batch**：如果单请求可以复现问题，在 batch_size=1 下排查。这样可以排除 batching 交互，seqlen 的形状也更简单。

### 策略：从 layer 0 开始

从第一个 linear attention block 或 full attention block 的 **layer 0** 开始，不要直接从中间层开始。原因：模型的大多数层使用相同的算子集合，layer 0 的情况具有代表性。如果 layer 0 没有问题，其他层大概率也没问题；如果 layer 0 已经有偏差，先修复它再往后看。

1. 在 layer 0 中，按顺序在每个子算子之后 dump：RMSNorm → Attention → MLP。
2. 与参考框架在同一层的结果对比（例如 Ascend 使用 vllm+vllm-ascend）。
   - 某个子算子在 layer 0 就有偏差 → 这就是第一个出现问题的算子，深入排查。
   - layer 0 所有子算子均正常 → 偏差在更深的层。对后续层使用二分查找（先看第 N/2 层，再缩小范围）。
3. 定位到有问题的算子后，有选择性地再验证一两个行为可能不同的层（例如最后一层、存在 MoE routing 的层）。

### 对比方法

**确定性 vendor 算子**（例如 Ascend 的 `torch_npu` 算子）：使用 `torch.equal()`。这类算子是确定性的，结果必须完全相同。任何差异都是真实 bug。

**非确定性算子**（例如 triton，Ascend 上较少见）：`torch.equal()` 可能因浮点舍入而过于严格，改用误差量来判断：

```python
diff = (a - b).abs()
print("最大绝对误差:", diff.max().item())
print("最大相对误差:", (diff / b.abs().clamp(min=1e-8)).max().item())
```

相对误差在 ~1e-3 以内通常可接受；超过这个量级则认为是真实偏差。

### 找到出现偏差的算子后

逐层读取其实现栈：
- `lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/` — `Impl` 类
- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/` — 薄 kernel wrapper
- `dlinfer/dlinfer/vendor/<vendor>/` — 实际硬件 op 调用（例如 Ascend：`ascend/torch_npu_ops.py`）

Dump 两个框架中该算子的**输入**，确认是否相同。若输入已经不同，则 bug 在上游；若输入相同但输出不同，则 bug 在算子调用本身（参数顺序错误、dtype 错误、shape 错误等）。

---

## 路径 C — 通信 / 并行策略

仅在某些并行配置下出现的精度问题，指向通信或并行策略 patch 的问题。排查前，先明确是哪个并行维度引入了问题。

### lmdeploy 并行术语

lmdeploy 支持三种并行维度的组合：

- **仅 TP**（EP=1, DP=1）：Attention 和 FFN 都在 `tp` 张 GPU 上切分。总 GPU 数 = tp。
- **dp×tp**（EP=1, DP>1）：Attention 总共使用 dp×tp 张 GPU；每个 DP 组内，`tp_size = tp / dp`。当 EP=1 时，配置中的 `tp` 等于总 GPU 数。
- **dp×tp + ep**（EP>1）：Attention 仍按 dp×tp 切分；FFN / MoE experts 进一步在 `ep` 组之间切分。当 EP>1 时，配置中的 `tp` 是**每个 DP 组的 tp_size**（不是总 GPU 数）。

### 隔离策略

不是所有模型都能在 TP=1 下运行。按并行复杂度从低到高逐步测试，在引入问题的那一层停下：

1. **仅 TP**（能装下权重的最简配置）：用最少 GPU 数的 TP-only 配置（DP=1, EP=1）运行两个框架。
   - 有问题 → 问题在 TP 算子切分或 all_reduce；结合路径 B 的 dump 方法，重点关注 all_reduce 前后的输出。
   - 正常 → 进入第 2 步。

2. **dp×tp**（加入 DP）：保持 EP=1，增大 DP。
   - 有问题 → 问题在 DP+TP 交互或 DP 组间通信；读取 `dlinfer/dlinfer/framework/lmdeploy_ext/device/<vendor>.py` 中相关的通信 patch。
   - 正常 → 进入第 3 步（仅限 MoE 模型）。

3. **dp×tp + ep**（加入 EP）：开启 EP>1。
   - 有问题 → 问题在 expert 并行或 EP 通信；读取 `device/<vendor>.py` 中的 MoE forward 类（例如 Ascend 的 `AscendMoEForwardDPTP`），核查 MoE routing 和 reduce-scatter 模式。

### 空闲 DP 组的 dummy 数据

当 dp > 1 时，lmdeploy 会在没有实际请求的 DP 组中填入**长度为 1 的 dummy 数据**，vllm-ascend 也有类似机制。这是预期行为，不是 bug。在跨 DP 组 dump tensor 时需注意：

- 空闲 DP 组的 tensor leading dimension 为 1——不要误认为是 seqlen 不一致。
- 只对实际处理了真实 token 的 DP 组做数值对比。
- 如果精度问题恰好出现在空闲组的 dummy 路径上，需确认两个框架使用了相同的 dummy 长度，且 dummy 数据没有污染真实组的 KV cache slot。

### 当 TP=1 装不下权重时

如果模型过大无法在 TP=1 下运行，从能加载权重的最小 TP 出发，在两侧使用相同的 TP。此时仍可以固定 TP、分别变化 DP 和 EP 来隔离各维度的影响。

### 需要读的文件

- `dlinfer/dlinfer/framework/lmdeploy_ext/device/<vendor>.py` — 该硬件的分布式行为 patch（例如 Ascend 的 `ascend.py`，包含 MoE 通信的 `AscendMoEForwardDPTP`）。
- 在每次 all_reduce / all_gather 调用的**前后**分别 dump 各 rank 的输出，找到第一次出现偏差的通信操作。

---

## Tensor dump 操作方法

**必须 dump 到文件，禁止使用 `print` 或 `logger`。**

在多 rank 场景下，所有 rank 的日志交错输出，tensor 数值会被冲掉无法读取。改用 `torch.save` 写入各 rank 独立的文件。

```python
import os, torch, torch.distributed as dist

_DUMP_DIR = "/tmp/dlinfer_dump"
os.makedirs(_DUMP_DIR, exist_ok=True)

def dump(name: str, tensor: torch.Tensor):
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.save(tensor.detach().cpu(), f"{_DUMP_DIR}/{name}_rank{rank}.pt")
```

**命名约定**：`{层编号}_{算子}_{input|output}_rank{rank}.pt`

示例：lmdeploy+dlinfer 的 `layer0_attn_out_rank0.pt`，vllm+vllm-ascend 的同名文件存放在不同目录下，方便配对比较。

**加载并比较**：

```python
a = torch.load("dlinfer/layer0_attn_out_rank0.pt")
b = torch.load("vllm/layer0_attn_out_rank0.pt")

# 确定性 vendor 算子（例如 Ascend 的 torch_npu）— 期望完全一致
print(torch.equal(a, b))

# triton / 浮点算子 — 检查误差幅度
diff = (a - b).abs()
print("最大绝对误差:", diff.max().item())
print("最大相对误差:", (diff / b.abs().clamp(min=1e-8)).max().item())
```

**dump 位置建议**：最好在 `lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/` 的 `Impl` 类中，紧接 kernel 调用之后、return 之前添加 dump。这一层位于 vendor-specific 代码之上，输出 tensor 的 shape 是框架原生格式，便于对比。

---

## 验收 checklist

- [ ] 相同 SoC 版本、关闭 warmup、`--eager-mode true`、相同 TP/DP/EP、temperature=0 / top_k=1 已确认
- [ ] 先尝试单 batch 复现
- [ ] 用同一 prompt 完成了输出 token 对比
- [ ] 从能装下权重的最简并行配置开始，逐步向上测试
- [ ] 确定了排查路径：A（KV cache）/ B（算子）/ C（通信）
- [ ] Tensor dump 使用文件方式（非 print / logger）
- [ ] 先验证了 layer 0，再向更深的层查找
- [ ] 找到第一个出现偏差的层 / 算子
- [ ] 已确认出现偏差的算子的输入是否相同（或发现上游 bug）
- [ ] 修复后重新验证输出 token 一致

---

## 故障排查

### 偏差随序列长度累积，但路径 A 没发现问题

**症状**：`kv_start_indices` 长度与 `key_states` 一致，但偏差仍随序列长度增加。

**原因**：算子精度问题（例如 cos/sin 回退到 CPU 计算）同样会在 decode 步骤中逐渐累积，
从外部看与 cache 污染完全相同。

**操作**：转入路径 B。Dump layer 0 的各子算子（RMSNorm → Attention → MLP），
确认偏差是否从这里开始。

---

### 某些 DP 组的 tensor leading dimension 出现意外的 1

**症状**：dp > 1 时，空闲 DP 组的 tensor 首维为 1，输出结果看起来异常。

**原因**：lmdeploy（以及 vllm-ascend）会在没有实际请求的 DP 组中填入长度为 1 的 dummy
数据，这是预期行为。

**操作**：只对实际处理了真实 token 的 DP 组做数值对比，不要将空闲组的 length-1 tensor
视为错误。

---

### dp×tp 或 ep 配置下有精度问题，但不清楚是哪个维度引入的

**症状**：加入 DP 或 EP 后出现精度回退，但无法确定是哪个维度导致的。

**原因**：同时改变多个并行维度，无法逐一排查。

**操作**：固定 TP，先单独加 DP；dp×tp 正常后再加 EP。详见路径 C 的隔离策略。

---

### dump 文件为空、内容截断或数值混乱

**症状**：保存的 dump 文件没有可用数据，或来自多个 rank 的数值混杂在一起。

**原因**：多 rank 场景下使用 `print` 或 `logger`，各 rank 的输出交错覆盖。

**操作**：使用 `torch.save` 为每个 rank 写独立文件。参见 Tensor dump 操作方法节。

---

### KV cache 污染 bug 偶现，无法稳定复现

**症状**：同一 prompt 多次运行结果不同，精度问题不稳定。

**原因**：warmup 在 KV cache 中遗留脏数据，影响后续推理。

**操作**：两侧都关闭 warmup，重新对比。

---

### 同一 prompt 每次生成的 token 不同

**症状**：token 级别对比不稳定，每次运行结果都变。

**原因**：temperature > 0 引入采样随机性，对比结果没有意义。

**操作**：两侧均设置 temperature=0、top_k=1。

---

### 二分查找结束后发现 bug 其实在很早的层

**症状**：经过多次二分，最终定位到的出问题层在模型很早的位置。

**原因**：从 layer N/2 开始跳过了对 layer 0 的检查。

**操作**：先验证 layer 0；如果 layer 0 正常，再对后续层二分。

---

### 找到疑似出问题的算子，输入看起来一致，但根因不明

**症状**：两个框架中该算子的输入相同，但无法判断算子本身是否有问题。

**原因**：没有 dump 输出，缺乏算子是否产生错误结果的直接证据。

**操作**：同时 dump 输入和输出。若输入一致但输出不同，bug 在算子调用本身——检查传给
硬件算子（例如 Ascend 的 NPU op）的参数顺序、dtype 和 shape。