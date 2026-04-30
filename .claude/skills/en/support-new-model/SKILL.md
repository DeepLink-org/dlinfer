---
name: support-new-model
description: Add support for a new model (already in lmdeploy's CUDA backend) on domestic AI hardware (Ascend / CAMB / MACA) via dlinfer.
---

You are helping the user adapt a new LLM or VLM for dlinfer's supported hardware backends
(Ascend NPU, CAMB MLU, MACA GPU). The model already runs on CUDA via lmdeploy — your job
is to identify what is missing for the target vendor and implement it.

---

## Step 1 — Gather information

Ask the user:
1. **Which model** are you adding support for? (name as it appears in `lmdeploy/pytorch/models/`, e.g. `qwen3`, `deepseek_v2`)
2. **Which vendor(s)** are you targeting? (ascend / camb / maca — may be multiple)

Do not proceed until both questions are answered.

---

## Step 2 — Analyse the model

Read all of the following files yourself using Read/Bash tools — do not ask the user:

```
lmdeploy/lmdeploy/pytorch/models/<model>.py
lmdeploy/lmdeploy/pytorch/backends/dlinfer/op_backend.py
lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py   ← one per target vendor
```

The full call chain is: `models/<model>.py` → `lmdeploy/pytorch/nn/` → `backends/dlinfer/`
→ `kernels/dlinfer/` → `dlinfer/ops/` → `vendor/`. If the connection between a model
layer and its backend op is unclear, trace through `lmdeploy/pytorch/nn/` to find the
intermediate abstraction. `lmdeploy/pytorch/kernels/default/` contains the CUDA reference
implementations and is useful as a specification when writing new vendor ops.

### From `models/<model>.py`, identify:

- Every non-trivial operator the model uses: attention variants (paged, flash, MLA),
  MLP activation functions, RMS norm variants, MoE routing, rotary embedding variants
  (standard, MROPE, multi-scale), quantization ops.
- Whether the model passes any fields through `StepContext` or `attn_metadata` beyond
  the standard set: `input_ids`, `position_ids`, `block_offsets`, `q_seqlens`,
  `kv_seqlens`, `kv_start_indices`.
  Known extra fields already handled: `state_ids` (SSM), `mrope_position_ids` (MROPE),
  `cu_seqlens` / `has_initial_state` (Gated Delta Networks).

### From the generic `op_backend.py`, check:

- `get_layer_impl_builder()`: which `OpType`s already have a dlinfer `Impl`.
  Cross-reference with the op list above to identify gaps → **Path A**.

### From `<vendor>/op_backend.py`, check each of the following carefully:

- **`update_step_context()`**: this method builds `attn_metadata` and (for Ascend)
  `moe_metadata` for every inference step. Verify that it correctly handles all fields
  the new model requires. If the model introduces new context fields or a new attention
  mode (e.g. a new `is_gated_delta`-style flag), this method must be extended → **Path B**.
- **`get_k_block_shape()` / `get_v_block_shape()`**: confirm the KV cache layout matches
  what the model's attention implementation expects. Different vendors and even different
  SoC generations (Ascend A2 vs A3, 310P) may use different layouts → **Path B** if wrong.
- **`AscendKVQuantMeta`** (Ascend only): if the model uses KV cache quantization with a
  scale/offset format different from the current implementation → **Path B**.

Summarise your findings to the user before writing any code:
- Op gaps (→ Path A)
- Vendor `op_backend.py` gaps (→ Path B)
- Framework-level gaps (→ Path C)

---

## Path A — Add missing ops (4-layer stack)

Follow this path for every op that is absent from `get_layer_impl_builder()`.

Implement each layer in top-to-bottom order:

### Layer 1 — `lmdeploy/lmdeploy/pytorch/backends/dlinfer/`

Add a new `XxxImpl` (inherits lmdeploy base `Impl`) and `XxxBuilder` (with `build()`).
Register the builder in `op_backend.py`'s `get_layer_impl_builder()` dispatcher.
Reference: `activation.py` (simplest), `norm.py`, `attention.py` (most complex).

### Layer 2 — `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/`

Add a thin wrapper function calling `dlinfer.ops.<op_name>(...)`.
Export it from `__init__.py`.

### Layer 3 — `dlinfer/dlinfer/ops/llm.py`

Register with `@register_custom_op("dlinfer::<op_name>", [...])`.
Forward to `vendor_ops_registry["<op_name>"]`.
**The key string must exactly match the function name in Layer 4.**

### Layer 4 — `dlinfer/dlinfer/vendor/<vendor>/`

Add `@register_ops(vendor_ops_registry)` implementation calling the vendor's native op:
- **Ascend**: `torch.ops.npu.*` — see `vendor/ascend/torch_npu_ops.py`
- **CAMB**: `tmo.*` (`torch_mlu_ops`) — see `vendor/camb/camb_ops.py`
- **MACA**: `mcoplib.*` — see `vendor/maca/maca_ops.py`

**Ascend**: before writing any new op in `torch_npu_ops.py`, ask the user to provide
the official NPU operator documentation for that op. Implement strictly according to
the docs: parameter names, tensor shapes, and dtype constraints are not always
inferrable from existing code and a mismatch causes hard-to-debug runtime errors.

For complex ops (e.g. Ascend attention with graph-mode bookkeeping), split logic into
a helper module (e.g. `vendor/ascend/attention.py`) and import from `torch_npu_ops.py`.

---

## Path B — Vendor-specific `op_backend.py` changes

File: `lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py`

Handle each sub-case independently:

### B1 — `update_step_context()`: new context fields or attention modes

When the new model requires fields in `attn_metadata` that the current implementation
does not populate, extend `update_step_context()`:
- Add the computation of the new field (following the existing helper-function pattern
  inside the method).
- Pass the new field when constructing `attn_metadata` at the end of the method.
- For Ascend: also extend `moe_metadata` if the model introduces a new MoE communication
  pattern or parallelism topology.

Reference: the `is_gated_delta` block (adds `cu_seqlens` and `has_initial_state`),
the `kv_quant_policy == 8` block (populates `AscendKVQuantMeta`).

### B2 — `get_k_block_shape()` / `get_v_block_shape()`: KV cache layout

This rarely needs changing once the hardware target is fixed. Skip unless the new model
introduces a fundamentally different attention architecture that requires a new block
memory layout not covered by any existing vendor backend.

### B3 — `AscendKVQuantMeta`: KV quantization (Ascend only)

Legacy feature; its correctness is not actively verified. Skip for standard model
support — only revisit if KV cache quantization is explicitly required and confirmed
to be working.

---

## Path C — Framework patches (`dlinfer/dlinfer/framework/lmdeploy_ext/`)

Each sub-area is independent — assess and handle separately.

### C1 — cudagraph / aclgraph buffer management

**When needed**: only when the model introduces a new `StepContext` field whose **shape
varies with batch size or sequence length** at runtime. Fixed-shape tensors do not need
special buffer management. Example: `x_active_mask` (shape `[batch_size]`) was added
to handle Expert Parallelism — its size changes per step, so it requires a pre-allocated
maximum-size buffer.

- **Ascend**: `framework/lmdeploy_ext/cudagraph/ascend_cudagraph.py`
  - `make_buffers_cudagraph`: allocate the new field at maximum size (`max_batches` /
    `max_tokens`). Using runtime size here causes shape errors on replay.
  - `fill_buffers_cudagraph`: copy runtime values into the pre-allocated buffer.
  - `update_context_cudagraph`: wire the buffer back into the step context.
  - Reference: `is_ssm` (`state_ids`) and `use_mrope` (`mrope_position_ids`) paths.
- **Other vendors**: apply the same pattern in `camb_cudagraph.py` / `maca_cudagraph.py`.

Skip if the model uses only the standard fields already handled.

### C2 — Device-specific patches

**When needed**: when the model requires a vendor-specific override of lmdeploy behaviour
(e.g. a different MoE communication strategy on Ascend, an unsupported sampling op on
CAMB, hardware-specific cache formats such as Ascend 310P NZ layout).

- **Ascend**: `framework/lmdeploy_ext/device/ascend.py`
- **CAMB**: `framework/lmdeploy_ext/device/camb.py`

Patch the relevant lmdeploy class method directly. Ensure the file is imported in
`framework/lmdeploy_ext/device/__init__.py`.

### C3 — Quantization patches

**When needed**: only when the model uses AWQ and the weight packing or scale layout
differs from the current Ascend implementation.

File: `framework/lmdeploy_ext/quants/ascend_awq.py`

This file patches `WeightOnlyQLinear`, `MergedAwqLinear`, `AwqLinear`, and
`QKVAwqLinear`. Only modify if the new model's quantized checkpoint uses a layout the
current patches cannot handle.

---

## Verification checklist

**Path A (new op):**
- [ ] All 4 layers implemented for each missing op
- [ ] `get_layer_impl_builder()` dispatcher updated in generic `op_backend.py`
- [ ] `vendor_ops_registry` key in `ops/llm.py` exactly matches the decorated function name in the vendor file
- [ ] New kernel exported from `kernels/dlinfer/__init__.py`

**Path B (vendor `op_backend.py`):**
- [ ] `update_step_context()` populates all fields the new model's `attn_metadata` requires

**Path C1 (graph buffers):**
- [ ] New field pre-allocated at max size in `make_buffers_cudagraph`
- [ ] New field filled in `fill_buffers_cudagraph`
- [ ] New field wired back in `update_context_cudagraph`

**Path C2 (device patch):**
- [ ] Patch applied directly to the lmdeploy class
- [ ] Patch file imported in `device/__init__.py`

**Path C3 (quant patch):**
- [ ] Weight packing / scale layout verified against checkpoint format
- [ ] Relevant class methods patched in `ascend_awq.py`

**General:**
- [ ] Eager mode: model runs without error
- [ ] Graph mode: model runs without error (if vendor supports it)
