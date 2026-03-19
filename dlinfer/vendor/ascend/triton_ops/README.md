# Triton Operators for Qwen3.5 Inference on Ascend NPU

This directory contains custom Triton kernels optimized for running Qwen3.5 (linear attention variant) inference on Huawei Ascend NPUs.

## Directory Structure

```
triton_ops/
├── __init__.py                    # Module entry point, exports all operators
├── causal_conv1d.py               # Causal 1D convolution (prefill + decode)
├── rms_norm_gated.py              # Gated RMSNorm with SiLU activation
├── triton_utils.py                # NPU device property helpers
└── fla/
    ├── __init__.py                # FLA submodule entry
    └── chunk.py                   # Chunked gated delta rule (prefill)
```

## Exported Operators

| Operator | File | Phase | Description |
|----------|------|-------|-------------|
| `causal_conv1d_fn` | `causal_conv1d.py` | Prefill | Varlen causal conv1d using PyTorch |
| `causal_conv1d_update_npu` | `causal_conv1d.py` | Decode | Stateful causal conv1d Triton kernel |
| `chunk_gated_delta_rule` | `fla/chunk.py` | Prefill | Chunked gated delta rule attention |
| `RMSNormGated` | `rms_norm_gated.py` | Both | Gated RMSNorm layer |

---

## Operator Details

### 1. `causal_conv1d_fn` & `causal_conv1d_update_npu`

**File**: `causal_conv1d.py`

**Source**:
- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/mamba/causal_conv1d.py
- Original: https://github.com/Dao-AILab/causal-conv1d

**Purpose**:
Implements causal 1D convolution for the convolutional path in Qwen3.5's linear attention mechanism. The causal nature ensures each output position only depends on current and past inputs (no future leakage).

**Two Functions**:
1. `causal_conv1d_fn` (Prefill): Pure PyTorch implementation using `F.conv1d`, handles variable-length sequences via `query_start_loc`
2. `causal_conv1d_update_npu` (Decode): Triton kernel for single-token or multi-token generation with state management

---

### 2. `chunk_gated_delta_rule`

**File**: `fla/chunk.py`

**Source**:
- Kernel: https://gitcode.com/Ascend/triton-ascend-kernels/blob/master/src/triton_ascend_kernels/attention/fla/
- Original: https://github.com/fla-org/flash-linear-attention (MIT)

**Purpose**:
Implements the core linear attention operation for prefill phase using chunked parallel computation. The "gated delta rule" is a variant of linear attention that:
- Uses gating to control information flow
- Applies delta rule for state updates (subtracting projected keys from values)
- Processes sequences in chunks for parallelism

**Wrapper Role**:
`fla/chunk.py` is a thin wrapper that handles input validation, dtype conversion, optional Q/K L2 normalization (`use_qk_l2norm_in_kernel`), and delegates to `triton_ascend_kernels.attention.fla.chunk_gated_delta_rule_fwd`.

---

### 3. `RMSNormGated`

**File**: `rms_norm_gated.py`

**Source**:
- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/layernorm_guard.py
- Original: https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py

**Purpose**:
Implements RMSNorm (Root Mean Square Normalization) with optional gated SiLU activation. The gating allows the norm output to be modulated by a "gate" tensor.

**Why Required**:
- Qwen3.5 uses gated normalization in its architecture
- Standard PyTorch normalization doesn't support gating
- Fused kernel is more efficient than separate norm + gate + mul

**Formula**:
- Without gate: `y = (x / sqrt(mean(x^2) + eps)) * weight`
- With gate: `y = norm(x) * silu(z)` (if `norm_before_gate=False`)

---

### 4. `triton_utils.py` (Helper)

**Source**:
- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/triton_utils.py

**Purpose**:
Provides NPU device property queries for Triton kernel tuning:
- `init_device_properties_triton()`: Initialize device properties
- `get_aicore_num()`: Get number of AI cores
- `get_vectorcore_num()`: Get number of vector cores

**Why Required**:
- Ascend NPUs have different core counts (AI cores vs vector cores)
- Some kernels use this information for optimal grid sizing
- Called during backend initialization

---

## Data Flow in Qwen3.5 Inference

### Prefill Phase
```
Input tokens
    ↓
[Causal Conv1D] ← causal_conv1d_fn
    ↓
[Linear Attention] ← chunk_gated_delta_rule (triton_ascend_kernels)
    ↓
[Gated RMSNorm] ← RMSNormGated
    ↓
Output logits
```

### Decode Phase
```
New token
    ↓
[Causal Conv1D Update] ← causal_conv1d_update_npu
    ↓
[Delta Rule Update] ← npu_recurrent_gated_delta_rule (torch_npu)
    ↓
[Gated RMSNorm] ← RMSNormGated
    ↓
Output logits
```

---

## External Dependencies

| Package | Usage |
|---------|-------|
| `triton` | Kernel DSL and JIT compilation |
| `triton_ascend_kernels` | Official Ascend-optimized kernels (prefill attention, recurrent update) |
| `torch` | Tensor operations, NPU device management |
| `torch_npu` | Ascend NPU backend for PyTorch |

---

## Installation

### Installing `triton-ascend-kernels`

The core attention kernels (`chunk_gated_delta_rule_fwd`) are provided by the official `triton-ascend-kernels` package.

**Prerequisites**:
- Python >= 3.8
- `triton-ascend` == 3.2.0
- `torch` == 2.6.0
- `torch_npu` == 2.6.0post3

**Install from source**:
```bash
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
cd triton-ascend-kernels
pip install -e .
```

**Verify installation**:
```python
from triton_ascend_kernels.attention.fla import chunk_gated_delta_rule_fwd
print("triton-ascend-kernels installed successfully!")
```

For more details, see: https://gitcode.com/Ascend/triton-ascend-kernels

---

## License

All kernels are Apache-2.0 licensed. Original sources are attributed in individual file headers.
