---
name: op-callchain
description: The operator call chain in lmdeploy+dlinfer — the five layers every op crosses from model.forward() down to torch_npu/aclnn kernels, and how to trace or add a new op. Uses fused MoE (Qwen3.5 MoE) on Ascend as the worked example.
---
# op-callchain

Every operator that a domestic backend (Ascend / CAMB / MACA) implements
crosses the **same five layers** in lmdeploy+dlinfer. This skill describes
that layering as a reusable map, so you can (a) trace where an existing op
goes, and (b) add a new op by filling in the same five slots. Fused MoE for
Qwen3.5 MoE on Ascend runs through as the worked example, but the structure is
identical for attention, RMS norm, rotary embedding, quantized matmul, etc.

Paths below are relative to the repo root and its sibling `../lmdeploy`.

---

## The five layers

```
1. modeling            lmdeploy/pytorch/models/<model>.py
2. nn abstraction      lmdeploy/pytorch/nn/<op>/            + backends/selector.py
3. backend dispatch    lmdeploy/pytorch/backends/dlinfer/{op_backend,<op>}.py
                       + <vendor>/op_backend.py  (per-step metadata)
4. dlinfer op iface    lmdeploy/pytorch/kernels/dlinfer/<op>.py  →  dlinfer/ops/llm.py
5. vendor kernel       dlinfer/vendor/<vendor>/…   (torch_npu / aclnn kernels)
```

Rules of thumb that hold for every op:

- **The model never names a device.** Layer 1 only calls an `nn` module.
- **Layer 2 decouples model from hardware** via `OpType` + `get_backend()`.
- **Layer 3 holds the per-step configuration** — the fields an op needs on
  `StepContext` / `attn_metadata` are populated in `<vendor>/op_backend.py`.
- **Layer 4 is a thin pass-through** into a global registry keyed by string.
- **Layer 5 is where real kernels live.** The *same* kernels run in eager and
  in graph mode — graph mode captures and replays them (see below).

The interesting work when adding an op is almost always concentrated at
**layer 3** (what config the op needs) and **layer 5** (which vendor kernels).

---

## Layer 1 — modeling

The model file calls an `nn` module, not a bare function.

> **MoE example** — `lmdeploy/pytorch/models/qwen3_5_moe.py`
> ```
> Qwen3_5MoeSparseMoeBlock.forward()                        # :108
>  ├─ self.gate(hidden_states)          → topk_weights, topk_ids
>  └─ self.experts(hidden_states, topk_weights, topk_ids)   # :115
> ```
> `self.experts` is a `FusedMoE` module built by `build_fused_moe()`
> (`nn/moe/__init__.py:11`). Routing (`self.gate`) is a *separate* op
> (`moe_gating_topk_softmax`) that shares the same metadata.

**To add an op:** find where the model calls it, and confirm it goes through an
`nn` module. If the model calls a raw torch function, that op is not yet
abstracted — you may need to add the `nn` module too.

---

## Layer 2 — nn abstraction + backend selection

The `nn` module resolves a device implementation in its `__init__` via an
`OpType`, then just calls `self.impl.forward(...)`.

```
<OpModule>.forward()  →  self.impl.forward(...)
# self.impl built in __init__:
#   impl_builder = get_backend().get_layer_impl_builder(OpType.<X>)
```

- `get_backend()` (`backends/selector.py:28`) reads the device context and, for
  `device_type == 'ascend'`, returns `AscendOpsBackend`.

> **MoE example** — `FusedMoE.forward()` (`nn/moe/default.py`) calls
> `self.impl.forward`; `impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)`.
> `build_fused_moe()` also selects quantized variants (`FusedMoEW8A8`,
> `FusedMoEBlockedF8`).

**To add an op:** ensure an `OpType.<X>` exists and the `nn` module requests it.

---

## Layer 3 — backend dispatch (dlinfer) + per-step metadata

`backends/dlinfer/op_backend.py` maps `OpType.<X>` → a builder, which builds an
`Impl` whose `forward()` calls the layer-4 kernel wrapper.

```
DlinferOpsBackend.get_layer_impl_builder(OpType.<X>)  →  Dlinfer<X>Builder
 └─ Dlinfer<X>Impl.forward(...)  →  <x>(...)   # the kernels/dlinfer wrapper
```

**This is where per-step configuration lives.** Anything the op needs beyond
its tensors is assembled in `<vendor>/op_backend.py:update_step_context()` and
read off `step_context` inside `Impl.forward`.

> **MoE example**
> `get_layer_impl_builder(OpType.FusedMoE)` → `DlinferFusedMoEBuilder`
> (`backends/dlinfer/op_backend.py:48`). `DlinferFusedMoEImpl.forward`
> (`backends/dlinfer/moe.py:73`) reads `step_context.moe_metadata`, built by
> `AscendOpsBackend.update_step_context()`
> (`backends/dlinfer/ascend/op_backend.py:496`), which sets:
> - `moe_comm_type` — **MC2 / ALLTOALL / ALLGATHER / naive**, chosen by token
>   count + SoC version. The main performance lever; selects the kernel path.
> - `x_active_mask` — padding mask for graph mode / MC2.
> - `moe_group_name`, tp/ep groups — HCCL groups for expert parallelism.
> - `router_n_groups` — grouped-gating switch read by the router op.

**To add an op:** if it needs extra state, add fields in `update_step_context`
and read them in `Impl.forward`. Known precedents: `state_ids` (SSM),
`mrope_position_ids` (MROPE), `cu_seqlens` (Gated Delta), `moe_metadata` (MoE).

---

## Layer 4 — dlinfer op interface

A thin wrapper in `kernels/dlinfer/<op>.py` calls `dlinfer.ops`, which
dispatches through the global `vendor_ops_registry` keyed by an **exact
string**.

```
kernels/dlinfer/<x>.py :  <x>(...)  →  ext_ops.<x>(...)      # ext_ops = dlinfer.ops
dlinfer/ops/llm.py     :  def <x>(...): return vendor_ops_registry["<x>"](...)
```

- The vendor module registers under `"<x>"` at import time
  (`dlinfer/vendor/__init__.py` builds the registry). This registry dispatch is
  the path actually used at runtime today, for both eager and graph mode.
- Ops are also (usually) exposed as a torch custom op via `@register_custom_op`
  (`dlinfer::<x>`). **This custom-op form is not exercised on the current
  runtime path** — graph mode is now capture/replay (see layer 5), not a
  torch.compile / dynamo trace, so nothing lowers `torch.ops.dlinfer.<x>`.
  Keep it anyway: **new ops should still add the `@register_custom_op`
  registration** so that `torch.compile`-based tracing can be re-enabled later
  without retrofitting every op. MoE happens to skip the eager custom-op form
  because of its complex argument types; prefer adding it for a new op unless
  the argument types make it impractical.

> **MoE example** — `kernels/dlinfer/fused_moe.py:8` → `dlinfer/ops/llm.py:617`
> → `vendor_ops_registry["fused_moe"]`. Router: `moe_gating_topk_softmax`
> (`dlinfer/ops/llm.py:496`).

**To add an op:** add the wrapper in `kernels/dlinfer/`, the interface function
in `dlinfer/ops/llm.py`, pick the registry key string, and add the
`@register_custom_op` registration (kept for future torch.compile).

---

## Layer 5 — vendor kernel

The vendor module registers the implementation with `@register_ops` and calls
concrete `torch.ops.npu.*` kernels. **There is one implementation, used by both
eager and graph mode.**

> **MoE example** — `dlinfer/vendor/ascend/torch_npu_ops.py:627` + `ascend/moe.py`
> ```
> fused_moe()
>  ├─ moe.moe_prepare()          # padding / mask / topk dtype
>  ├─ branch on moe_comm_type:
>  │   ├─ MC2      → fused_moe_mc2()   npu_moe_distribute_dispatch_v2 / _combine_v2
>  │   ├─ ALLTOALL → fused_moe_all2all() npu_moe_token_permute / _unpermute
>  │   └─ naive    → fused_moe_naive()
>  │                  npu_moe_init_routing_v2   # scatter tokens to experts
>  │                  npu_grouped_matmul        # gate_up projection
>  │                  npu_swiglu                # activation
>  │                  npu_grouped_matmul        # down projection
>  │                  npu_moe_token_unpermute   # gather back
>  └─ moe.moe_finalize()
> ```
> Router: `npu_moe_gating_top_k` if `router_n_groups > 0`, else
> `npu_moe_gating_top_k_softmax`.

### Eager vs graph — same kernels, capture/replay

Graph mode is **not** a separate lowering of the op. It follows a
cudagraph-style route: the exact same layer-5 kernels are recorded once
(`torch.npu.NPUGraph`, decode only) and replayed on later steps with fixed
input/output buffers. There is no ATB IR and no per-op graph conversion.

The only op-author concern is what NPUGraph **cannot** capture and therefore
must be **isolated from the eager path**:

- **Dynamic shapes** — capture is keyed by a bucketed batch size; variable
  tensors must be backed by fixed-shape buffers.
- **Host-side / CPU parameters** — values a kernel reads from the host (not
  from an NPU input buffer) are frozen at capture time and must be pushed in
  via an explicit update API on replay.
- **Data-dependent control flow / Python-side branching** — anything that
  changes across steps cannot live inside the captured region.

These concerns are handled by the graph runner and the three buffer methods,
not inside the kernel. See **[[graph-mode-internals]]** for the runner
architecture, buffer management, and the Ascend `kv_seqlens` update mechanism.
`x_active_mask` (layer 3) is one example of state added specifically so MoE
behaves correctly under capture.

**To add an op:** implement the eager path under `dlinfer/vendor/<vendor>/`
(register with `@register_ops`). If the op runs during decode under graph mode,
make sure any dynamic/host-side inputs are buffer-backed or updated through the
graph runner rather than captured as constants.

---

## Checklist — adding a new op

1. **Layer 1** — model calls an `nn` module (add the module if it doesn't exist).
2. **Layer 2** — define/confirm `OpType.<X>`; `nn` module requests it via
   `get_backend().get_layer_impl_builder`.
3. **Layer 3** — add `Dlinfer<X>Builder`/`Impl` in `backends/dlinfer/<x>.py`,
   wire it in `op_backend.py:get_layer_impl_builder`; add any per-step fields in
   `<vendor>/op_backend.py:update_step_context`.
4. **Layer 4** — wrapper in `kernels/dlinfer/<x>.py`, interface in
   `dlinfer/ops/llm.py`, choose the `vendor_ops_registry` key, and add the
   `@register_custom_op` registration (unused now, kept for future torch.compile).
5. **Layer 5** — eager impl in `dlinfer/vendor/<vendor>/` (`@register_ops`).
   For graph mode, no separate conversion is needed; instead ensure dynamic /
   host-side inputs are buffer-backed or updated via the graph runner.
6. **Reference** — `lmdeploy/pytorch/kernels/default/` holds the CUDA reference
   implementation; use it as the correctness spec, and eager mode as the
   runtime reference (see [[graph-mode-internals]] and [[precision-align]]).

## Points to note

1. **Registry key strings must match exactly.** A vendor that registers under a
   different string than `dlinfer/ops/llm.py` looks up silently falls through.
2. **Per-step config is layer 3, not the kernel.** Perf/behaviour switches like
   MoE's `moe_comm_type` are decided in `update_step_context`; start debugging
   there, not in the vendor kernel.
3. **One model op can be several dlinfer ops.** MoE = router
   (`moe_gating_topk_softmax`) + experts (`fused_moe`) sharing one metadata.
   Don't assume a 1:1 mapping.
4. **Eager is the reference path.** If graph mode is wrong, run the step eager
   to localise the bug to capture/replay vs the underlying kernel.
5. **Graph mode is capture/replay, not a separate impl.** The same layer-5
   kernel runs both ways; graph correctness bugs are almost always about
   dynamic/host-side state that wasn't isolated from capture — look at the
   buffer methods and the graph runner, not for a second kernel.
6. **Keep `@register_custom_op` on new ops even though it's currently dormant.**
   It costs nothing now and is what a future `torch.compile` path would rely on.
