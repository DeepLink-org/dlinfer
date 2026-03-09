# triton-ascend-kernels 安装指南

本文档介绍如何安装 `triton-ascend-kernels`，这是运行 Qwen3.5 线性注意力模型所需的官方 Triton 算子库。

## 简介

`triton-ascend-kernels` 是基于 [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend) 的高性能 Triton 算子实现，专门为华为昇腾 NPU 优化。

**项目地址**: https://gitcode.com/Ascend/triton-ascend-kernels

## 依赖要求

| 依赖包 | 版本要求 |
|--------|----------|
| Python | >= 3.8 |
| triton-ascend | == 3.2.0 |
| torch | == 2.6.0 |
| torch_npu | == 2.6.0post3 |

## 安装步骤

### 方法一：从源码安装（推荐）

```bash
# 克隆仓库
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git

# 进入目录
cd triton-ascend-kernels

# 以可编辑模式安装
pip install -e .
```

### 方法二：开发模式安装（包含测试依赖）

```bash
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
cd triton-ascend-kernels
pip install -e ".[dev]"
```

## 验证安装

安装完成后，可以通过以下代码验证：

```python
# 验证 prefill 阶段的算子
from triton_ascend_kernels.attention.fla import chunk_gated_delta_rule_fwd
print("chunk_gated_delta_rule_fwd 导入成功!")

# 验证 decode 阶段的算子
from triton_ascend_kernels.moe.fused_recurrent import fused_recurrent_gated_delta_rule
print("fused_recurrent_gated_delta_rule 导入成功!")

print("triton-ascend-kernels 安装成功!")
```

## 在 dlinfer 中使用

dlinfer 的 `triton_ops` 模块依赖 `triton_ascend_kernels` 提供以下核心算子：

| 算子 | 来源 | 用途 |
|------|------|------|
| `chunk_gated_delta_rule_fwd` | `triton_ascend_kernels.attention.fla` | Prefill 阶段分块注意力 |
| `fused_recurrent_gated_delta_rule` | `triton_ascend_kernels.moe.fused_recurrent` | Decode 阶段循环更新 |

## 常见问题

### Q: 安装时报错 "triton-ascend not found"

确保已正确安装 `triton-ascend`:
```bash
pip install triton-ascend==3.2.0
```

### Q: 导入时报错 "No module named 'triton_ascend_kernels'"

检查是否在正确的 Python 环境中安装，并确认安装过程没有报错。

### Q: 版本不兼容

请确保 `torch` 和 `torch_npu` 版本与 `triton-ascend-kernels` 的要求一致。

## 参考资料

- [triton-ascend-kernels 仓库](https://gitcode.com/Ascend/triton-ascend-kernels)
- [Triton-Ascend 仓库](https://gitcode.com/Ascend/triton-ascend)
- [dlinfer triton_ops 文档](../dlinfer/vendor/ascend/triton_ops/README.md)
