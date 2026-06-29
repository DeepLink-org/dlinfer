---
name: graph-mode-internals
description: Understand the complete graph mode flow in lmdeploy+dlinfer, covering the runner architecture, buffer management, vendor differences, and common pitfalls.
---
# graph-mode-internals

This skill explains how graph mode works end-to-end in lmdeploy+dlinfer,
covering the runner layer, buffer layer, capture/replay flow, and
vendor-specific differences. The goal is understanding, not just
implementation details.

---

## Background

**What is graph mode?**
Graph mode captures a sequence of compute operations as a static graph and
replays it without Python overhead. In practice this means each decode step
can reuse a pre-compiled execution plan, reducing per-step latency.

**Why decode only — not prefill?**
Prefill sequence lengths vary widely across requests. Capturing a separate
graph for each possible length would require far too many buckets, consuming
large amounts of compile time and device memory. Decode is different: each
request generates exactly one new token per step, so `q_seqlen = 1` for all
requests. This makes bucketing by batch size alone practical.

**Eager mode** skips graph capture entirely and runs ops directly through
Python dispatch. It is the reference execution path.

---

## Code Organisation

### lmdeploy (base classes and CUDA implementation)

- **`CudaGraphMeta`** (`lmdeploy/pytorch/models/utils/cudagraph.py`) —
  dataclass that stores graph configuration: `max_batchs`, `max_tokens`,
  `num_blocks`, `device`, `input_buffers`, `output_buffers`, and optional
  flags for MLA, SSM, MRoPE, etc.
- **`CudaGraphMixin`** (same file) — mixin class that defines five methods
  with default CUDA implementations:
  - `support_cuda_graph` — returns True if the current step should use graph
    mode (default: True when decoding)
  - `make_buffers_cudagraph` — allocates fixed-shape tensors that will serve
    as graph inputs for all future replays
  - `fill_buffers_cudagraph` — copies real per-step data into the fixed
    buffers before capture or replay
  - `update_context_cudagraph` — updates `StepContext` fields to point at
    the buffer tensors
  - `get_outputs_cudagraph` — slices the full output buffers to the actual
    token count after replay
- **`GraphRunner`** (`lmdeploy/pytorch/backends/graph_runner.py`) — base
  class; `__call__` simply calls `self.model(**kwargs)` (no graph)
- **`CUDAGraphRunner`** (`lmdeploy/pytorch/backends/cuda/graph_runner.py`)
  — full CUDA implementation with `CUDASingleGraphRunner` (uses
  `torch.cuda.CUDAGraph`) and batch-size bucketing

### dlinfer (vendor extensions)

All vendors monkey-patch the three buffer methods at import time:

```python
CudaGraphMixin.make_buffers_cudagraph  = Vendor_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph  = Vendor_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = Vendor_update_context_cudagraph
```

Ascend additionally provides **`AscendGraphRunner`**, which extends
`GraphRunner` with `AscendSingleGraphRunner` (uses `torch.npu.NPUGraph`).
Camb, MACA, and PPU reuse lmdeploy's `CUDAGraphRunner`.

The wiring point where each vendor selects its runner class is
**`op_backend.build_graph_runner()`** in
`lmdeploy/pytorch/backends/dlinfer/<vendor>/op_backend.py`.

---

## Runner Layer

### Batch-size bucketing (`compatible_size`)

Graph capture is keyed by batch size. To maximise graph reuse, the actual
batch size is rounded up to the nearest bucket before looking up or
creating a graph:

- **Ascend** (`AscendGraphRunner.get_ascend_compatible_size`):
  three stages — power-of-2 for ≤ 16, 16-aligned for ≤ 256, 256-aligned
  for > 256
- **Camb / MACA / PPU** (via `CUDAGraphRunner`): pure power-of-2

### `_runner_map` and graph lifecycle

`_runner_map` maps `(compatible_batch_size, is_decoding, ...)` to a single
graph runner. On first encounter the runner captures the graph; on
subsequent encounters it replays the cached graph.

---

## Buffer Layer

### Two categories of tensors

| Category | Shape changes with batch size? | Needs buffer? |
|---|---|---|
| KV cache (`past_key_values`) | No — allocated once at max size | No |
| `q_seqlens`, `kv_seqlens`, `block_offsets`, … | Yes | Yes |

KV cache is passed through unchanged. Variable-shape tensors must be backed
by fixed-shape buffers so the captured graph always sees the same memory
addresses and shapes.

### The three buffer methods

**`make_buffers_cudagraph`** — called once during graph capture setup.
Allocates fixed-shape tensors on device (at `max_batchs` / `max_tokens`
size) and stores them in `graph_meta.input_buffers`.

**`fill_buffers_cudagraph`** — called before every capture and every
replay. Copies real data from the actual forward inputs into the
pre-allocated buffers. Pads unused slots with safe defaults (e.g. repeating
`max_tokens // max_batchs` for padding seqlens; initialising `kv_start_indices`
to -1 so that padding slots never corrupt KV cache slot 0).

**`update_context_cudagraph`** — called before every capture and replay.
Updates `StepContext` to point at the buffer tensors so that downstream ops
(e.g. attention) read from the right memory.

If you introduce a new tensor input that varies with batch size, all three
methods must be updated in sync.

---

## Capture Flow

```text
GraphRunner.__call__
  └─ compatible_size = get_compatible_size(batch_size)
       └─ _runner_map[compatible_size] not found → create AscendSingleGraphRunner
            (or CUDASingleGraphRunner for Camb / MACA / PPU)
            │
            ├─ make_buffers_cudagraph(graph_meta)  ← allocate fixed buffers once
            │
            ├─ fill_buffers_cudagraph(...)          ← copy real data into buffers
            │
            ├─ update_context_cudagraph(...)        ← point StepContext at buffers
            │
            ├─ warmup forward (outside graph scope)
            │
            └─ with torch.cuda.graph() / torch.npu.NPUGraph():
                 model.forward(...)                 ← ops captured here
                 make_output_buffers(output)        ← store output tensor refs
```

---

## Replay Flow

```text
GraphRunner.__call__
  └─ compatible_size = get_compatible_size(batch_size)
       └─ _runner_map[compatible_size] found → AscendSingleGraphRunner.forward()
            │
            ├─ fill_buffers_cudagraph(...)     ← update buffer contents
            │
            ├─ update_context_cudagraph(...)   ← re-point StepContext
            │
            ├─ [Ascend only] update kv_seqlens in-place (see next section)
            │
            ├─ _graph.replay()                 ← execute captured ops
            │
            └─ get_outputs_cudagraph(...)      ← slice output to actual token count
```

> **Note**: `get_outputs_cudagraph` is a simple output-slicing step. It
> reads `output_buffers['hidden_states']` and slices `[:, :num_tokens]`.
> For most vendors this is identical to the lmdeploy default.

---

## Ascend — kv_seqlens Update During Replay

For Camb and MACA, writing updated values into the input buffer before
replay is sufficient — the graph reads from the live device buffer automatically.
Ascend is different: the attention operator takes `actual_seq_lengths_kv`
as a CPU tensor or list, not as part of the NPU input buffer. An NPU buffer
write cannot reach this CPU-side parameter, so the new values must be
explicitly pushed into the captured graph via a dedicated update API.

Two mechanisms exist, selected at runtime by `aclgraph_use_torch_npu_update()`:

**torch_npu < 2.8.0.post1** — uses the low-level ACL graph task update API:

```python
graph_task_update_begin(graph_handle)
update_attn_params(kv_seqlens, ...)  # writes via ACL
graph_task_update_end(graph_handle)
```

**torch_npu ≥ 2.8.0.post1** — uses the higher-level torch_npu graph update
API:

```python
graph.update(cpu_update_input=[{"actual_seq_lengths_kv": kv_seqlens}])
```

---

## Vendor Comparison

| Item | Ascend | Camb | MACA |
|---|---|---|---|
| Runner | `AscendGraphRunner` | `CUDAGraphRunner` | `CUDAGraphRunner` |
| Graph API | `npu.NPUGraph` | `cuda.CUDAGraph` | `cuda.CUDAGraph` |
| `compatible_size` | 3-stage (p2/16-align/256-align) | power-of-2 | power-of-2 |
| `attn_metadata` slicing | not sliced | sliced | sliced |
| `kv_start_indices` | `(max_batchs,)` | `(max_batchs,)` | `(max_batchs, 1)` |
| `max_kv_seq_len` | kept as-is | set to -1 | kept as-is |
| `x_active_mask` buffer | Yes | No | No |
| kv_seqlens update | `update_attn_params` / `graph.update()` | write | write |

---

## Points to Note

1. **`kv_start_indices` must be initialised to -1, not 0.** Index 0 is a
   valid KV cache slot; padding slots initialised to 0 will silently corrupt
   it.

2. **`max_kv_seq_len` must be -1 for Camb.** This integer is captured as
   a constant node in the graph at capture time. The `torch_mlu_ops` API
   treats any value ≤ 0 as "compute the max dynamically from `kv_seqlens`";
   setting it to the actual max at capture time would make it wrong at
   every subsequent replay step.

3. **All three buffer methods must be updated together.** If you add a new
   tensor that varies with batch size, `make_buffers` must allocate the
   buffer, `fill_buffers` must copy data into it, and `update_context` must
   point `StepContext` at it. Missing any one of the three will cause
   incorrect behaviour or a silent read from stale data.

4. **Graph capture happens at `compatible_size`, not at actual batch size.**
   Batch sizes are rounded up to a bucket. Do not compare `new_batch_size`
   directly to `max_batchs` — use the compatible-size logic instead.

5. **Ascend kv_seqlens update version check.** When debugging Ascend graph
   mode failures involving wrong attention outputs, check which torch_npu
   version is in use and verify the correct update path is taken in
   `AscendSingleGraphRunner`.

6. **Eager mode is always available as a reference.** If graph mode produces
   wrong outputs, run the same step in eager mode (`eager_mode=True`) to
   confirm whether the bug is in graph capture/replay or in the underlying
   ops.
