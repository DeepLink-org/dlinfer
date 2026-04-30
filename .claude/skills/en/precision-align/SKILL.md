---
name: precision-align
description: Debug precision regressions in lmdeploy+dlinfer on domestic AI hardware (Ascend / CAMB / MACA) by comparing against a reference implementation.
---

You are helping the user fix a precision bug in lmdeploy+dlinfer on a domestic AI
hardware backend (Ascend, CAMB, or MACA). The reference implementation is typically
vllm+vllm-ascend (for Ascend) or another agreed-upon reference. The goal is to
identify where lmdeploy+dlinfer diverges from the reference and fix it.

The examples in this skill use Ascend as the concrete hardware. Apply the same
methodology to CAMB and MACA by substituting the appropriate vendor paths and ops.

---

## Step 1 — Gather information

Ask the user:

1. **Which model** are you aligning? (e.g., `qwen3`, `deepseek_v2`)
2. **Which hardware** are you targeting? (ascend / camb / maca)
3. **What is the symptom?** — e.g., output tokens differ from the first token, answers
   become nonsensical after a few tokens, accuracy benchmark score dropped by X points.
4. **Parallelism configuration**: what TP / DP / EP values are you using?
5. **Any preliminary observations?** First token already wrong (→ prefill issue), or
   diverges after a few correct tokens (→ decode / KV cache)?
6. **Single-batch or multi-batch?** Can you reproduce the issue with a single request
   (batch_size=1), or does it only appear when multiple requests are batched together?

Do not proceed until these are answered.

---

## Step 2 — Verify environment setup

Before any debugging, confirm the comparison environment is controlled. Both sides must
be identical except for the framework under test:

| Condition                        | lmdeploy+dlinfer          | vllm+vllm-ascend       |
|----------------------------------|---------------------------|------------------------|
| Same SoC version                 | ✓                         | ✓                      |
| Warmup disabled                  | ✓                         | ✓                      |
| Eager mode                       | ✓ (`--eager-mode true`)   | ✓ (`--enforce-eager`)  |
| Same TP / DP / EP                | ✓                         | ✓                      |
| `temperature=0`, `top_k=1`       | ✓                         | ✓                      |
| Same prompt / input              | ✓                         | ✓                      |

If any condition is unmet, fix it first. Warmup leaves stale KV cache entries;
temperature > 0 introduces sampling randomness — both mask real precision bugs.

---

## Step 3 — Quick output comparison

Run both frameworks on the **same prompt** and compare the generated tokens directly.

- **Tokens match** → output is consistent; precision is likely fine. Suggest running
  opencompass or evascope for benchmark scoring.
- **Tokens differ** → proceed to Step 4.

---

## Step 4 — Diagnose root cause

Map the symptom to a debugging path:

| Symptom                                              | Most likely cause                      | Path  |
|------------------------------------------------------|----------------------------------------|-------|
| First token already wrong                            | Prefill operator precision             | B     |
| First token correct, then diverges                   | KV cache pollution or decode op        | A or B|
| Divergence grows with sequence length                | KV cache pollution **or** op precision | A or B|
| Divergence appears immediately at a fixed depth      | Operator precision                     | B     |
| Only wrong at TP > 1 (or only at dp×tp / ep config) | Communication / parallelism patch      | C     |
| Only wrong with multiple requests, single is fine   | Batching / seqlen / masking issue      | A or B|

**Important**: accumulating divergence does **not** imply KV cache pollution. An
operator precision bug can also compound over decode steps — for example, a rope
embedding op silently falling back to CPU produces slightly wrong position encodings
that accumulate into a visible accuracy drop (observed on Qwen30B-A3B: a 2-point drop
on LiveCodeBench traced to cos/sin computed on CPU). Do not assume Path A without
ruling out Path B first.

**If unsure**:
- Start with a single-batch request to eliminate batching interactions.
- Start with the simplest parallelism configuration that can load the model weights
  (see Path C for the parallelism hierarchy). Some large models cannot fit at TP=1,
  so "simplest" means fewest parallelism dimensions that still fits the weights.
- Then go to Path B starting from layer 0.

---

## Path A — KV cache pollution

KV cache pollution means `fill_kv_cache` was called with mismatched indices, writing
tokens into wrong cache slots (stomping). The `fill_kv_cache` kernel itself is
generally not the source of bugs — the problem is almost always in the indices passed
to it.

### What to check

Read these two files:
- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/fill_kv_cache.py`
- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/pagedattention.py`

Key parameters:
- `kv_start_indices` in `fill_kv_cache`: flat index of each token's cache slot.
- `block_offsets`, `q_start_loc`, `q_seq_len`, `kv_seq_len` in `prefill_attention`
  and `paged_token_attention` / `paged_attention_fwd`.

**Multi-batch note**: if the bug only appears with multiple requests, pay extra
attention to per-request seqlen tracking (`q_seq_len`, `kv_seq_len`) and
`kv_start_indices`. A wrong per-request length causes attention to read from the
wrong positions in the KV cache.

### How to debug

Do **not** dump the KV cache tensors — they are prohibitively large. Instead, dump
the three tensors immediately **before** the `fill_kv_cache` call at the suspect layer:

```python
dump("key_states",       key_states)        # shape: [num_tokens, num_kv_heads, head_size]
dump("value_states",     value_states)      # shape: [num_tokens, num_kv_heads, head_size]
dump("kv_start_indices", kv_start_indices)  # shape: [num_tokens]
```

**Key check**: `kv_start_indices.shape[0]` must equal `key_states.shape[0]`. A
mismatch means the index count does not match the token count being written, which
causes fill-time stomping and corrupts subsequent decode steps.

---

## Path B — Operator precision

The goal is to find the **first op** where lmdeploy+dlinfer diverges from the reference.

**Single-batch first**: if the issue is reproducible with a single request, debug
at batch_size=1. This eliminates batching interactions and simplifies seqlen shapes.

### Strategy: start at layer 0

Start at **layer 0** of the first linear-attention or full-attention block. Do not
start at the midpoint: most of the model's layers share the same operator set, so
layer 0's result is representative. If layer 0 is clean, most other layers will be
too; if layer 0 already diverges, fix it before searching deeper.

1. At layer 0, dump after each sub-op in order: RMSNorm → Attention → MLP.
2. Compare with the reference framework at the same layer (e.g. vllm+vllm-ascend for Ascend).
   - Sub-op diverges at layer 0 → that is the first divergent op; investigate it.
   - Layer 0 is fully clean → use binary search across later layers (check layer
     N/2, then narrow down) to find the first divergent layer.
3. After identifying the faulty op, selectively verify one or two more layers that
   might behave differently (e.g. the last layer, MoE layers if applicable).

### Comparison method

For **deterministic vendor ops** (e.g. `torch_npu` on Ascend): use `torch.equal()`. These ops must
produce bit-identical outputs. Any difference is a real bug.

For **non-deterministic ops** (e.g. triton, less common on Ascend): `torch.equal()` may be too strict due to
FP rounding. Check error magnitude instead:

```python
diff = (a - b).abs()
print("max abs:", diff.max().item())
print("max rel:", (diff / b.abs().clamp(min=1e-8)).max().item())
```

A relative error below ~1e-3 is generally acceptable; above that it is a real
divergence.

### Once the divergent op is found

Read its implementation stack:
- `lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/` — the `Impl` class
- `lmdeploy/lmdeploy/pytorch/kernels/dlinfer/` — the thin kernel wrapper
- `dlinfer/dlinfer/vendor/<vendor>/` — the actual hardware op call (e.g. for Ascend: `ascend/torch_npu_ops.py`)

Dump the **inputs** to that op in both frameworks and check whether they are
identical. If inputs differ, the bug is upstream; if inputs are identical but outputs
differ, the bug is in the op itself (wrong argument order, dtype, or shape).

---

## Path C — Communication / parallelism

Precision bugs that only appear with certain parallelism configurations point to
communication or parallelism-patch issues. Before debugging, understand which
parallelism level introduces the problem.

### lmdeploy parallelism terminology

lmdeploy supports three parallelism dimensions used in combination:

- **TP only** (EP=1, DP=1): attention and FFN are both sharded across `tp` GPUs.
  Total GPUs = tp.
- **dp×tp** (EP=1, DP>1): attention uses dp×tp GPUs total; within each DP group,
  `tp_size = tp / dp`. When EP=1, `tp` in the config equals the total GPU count.
- **dp×tp + ep** (EP>1): attention uses dp×tp as above; FFN/MoE experts are further
  sharded across `ep` groups. When EP>1, `tp` in the config is the tp_size **per DP
  group** (not the total GPU count).

### Isolation strategy

Not all models fit at TP=1. Work through the parallelism hierarchy from simplest to
most complex, stopping at the level that introduces the bug:

1. **TP only** (simplest that fits the weights): run both frameworks with TP=N,
   DP=1, EP=1 (where N is the minimum number of GPUs needed to load the model).
   - Bug present → issue is in TP operator sharding or all_reduce; go to operator
     dumps (Path B) focusing on the all_reduce outputs.
   - Clean → proceed to step 2.

2. **dp×tp** (add DP): increase DP while keeping EP=1.
   - Bug appears → issue is in DP+TP interaction; check communication between DP
     groups. Read `dlinfer/dlinfer/framework/lmdeploy_ext/device/<vendor>.py` for
     relevant patches.
   - Clean → proceed to step 3 (only for MoE models).

3. **dp×tp + ep** (add EP): enable EP>1.
   - Bug appears → issue is in expert parallelism or EP communication. Read
     the MoE forward class in `device/<vendor>.py` (e.g. `AscendMoEForwardDPTP` in
     `device/ascend.py`) and verify the MoE routing and reduce-scatter pattern.

### Dummy data in idle DP groups

When dp > 1, lmdeploy fills DP groups that have no real requests with **dummy data
of sequence length 1**. vllm-ascend uses a similar mechanism. This is expected
behaviour, not a bug. When dumping tensors across DP groups:

- Idle DP groups will show tensors with a leading dimension of 1 — do not mistake
  this for a seqlen mismatch.
- Only compare tensor values in DP groups that are actually processing real tokens.
- If a precision discrepancy appears specifically in the idle-group dummy path,
  verify that both frameworks use the same dummy length and that the dummy data does
  not pollute the real groups' KV cache slots.

### When TP=1 is impossible

If the model is too large to fit at TP=1, start at the minimum TP that loads the
weights and compare it against the same TP on the reference side. You can still
isolate DP and EP by fixing TP and varying DP/EP independently.

### What to read

- `dlinfer/dlinfer/framework/lmdeploy_ext/device/<vendor>.py` — patches for
  distributed behaviours specific to the hardware (e.g. for Ascend: `ascend.py`
  containing `AscendMoEForwardDPTP` for MoE communication).
- Dump outputs immediately **before and after** each all_reduce / all_gather call
  across ranks to find where values first diverge.

---

## Tensor dump mechanics

**Always dump to files. Never use `print` or `logger`.**

On multi-rank runs, log output from all ranks interleaves and individual tensor values
are lost. Use `torch.save` to per-rank files instead.

```python
import os, torch, torch.distributed as dist

_DUMP_DIR = "/tmp/dlinfer_dump"
os.makedirs(_DUMP_DIR, exist_ok=True)

def dump(name: str, tensor: torch.Tensor):
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.save(tensor.detach().cpu(), f"{_DUMP_DIR}/{name}_rank{rank}.pt")
```

**Naming convention**: `{layer}_{op}_{input|output}_rank{rank}.pt`

Example: `layer0_attn_out_rank0.pt` for lmdeploy+dlinfer, same name in a separate
directory for vllm+vllm-ascend, so files are easy to pair.

**Loading and comparing**:

```python
a = torch.load("dlinfer/layer0_attn_out_rank0.pt")
b = torch.load("vllm/layer0_attn_out_rank0.pt")

# deterministic vendor ops (e.g. torch_npu on Ascend) — expect exact match
print(torch.equal(a, b))

# triton / float ops — check error magnitude
diff = (a - b).abs()
print("max abs:", diff.max().item())
print("max rel:", (diff / b.abs().clamp(min=1e-8)).max().item())
```

**Placement of dump calls**: add dumps inside the
`lmdeploy/lmdeploy/pytorch/backends/dlinfer/<vendor>/` `Impl` classes — right after
the kernel call and before returning. This gives the output in framework-native shape
and is above the vendor-specific layer.

---

## Checklist

- [ ] Same SoC, warmup disabled, `--eager-mode true`, same TP/DP/EP, temperature=0 / top_k=1 confirmed
- [ ] Single-batch reproduction attempted first
- [ ] Output token comparison done on the same prompt
- [ ] Parallelism hierarchy tested from simplest fitting config upward
- [ ] Root cause path identified: A (KV cache) / B (operator) / C (communication)
- [ ] Tensor dumps use file-based approach (not print / logger)
- [ ] Layer 0 verified first before searching deeper layers
- [ ] First divergent layer / op identified
- [ ] Inputs to the divergent op verified (identical or upstream bug found)
- [ ] Root cause fixed and output tokens re-verified

---

## Troubleshooting

### Accumulating divergence, but Path A checks out

**Symptom**: `kv_start_indices` length matches `key_states`, but divergence still
grows with sequence length.

**Cause**: Operator precision bugs (e.g. cos/sin falling back to CPU) also accumulate
across decode steps and look identical to cache pollution from the outside.

**Action**: Shift to Path B. Dump layer 0 sub-ops (RMSNorm → Attention → MLP) to
confirm whether divergence begins there.

---

### Tensors in some DP groups show unexpected length-1 shapes

**Symptom**: When dp > 1, tensors in idle DP groups have a leading dimension of 1 and
produce odd-looking outputs.

**Cause**: lmdeploy (and vllm-ascend) fill DP groups that have no real requests with
dummy data of sequence length 1. This is expected behaviour.

**Action**: Only compare tensor values in DP groups that are actually processing real
tokens. Do not treat length-1 tensors from idle groups as errors.

---

### Precision issue at dp×tp or ep, but unclear which dimension causes it

**Symptom**: Adding DP or EP introduces a precision regression, but the root cause
dimension is unknown.

**Cause**: Testing multiple parallelism dimensions simultaneously makes it impossible
to isolate which one introduces the bug.

**Action**: Fix TP, add DP first. If dp×tp is clean, then add EP. See Path C for the
full isolation strategy.

---

### Dump files are empty, truncated, or contain garbled content

**Symptom**: Saved dump files have no usable data, or values from multiple ranks are
mixed together.

**Cause**: Using `print` or `logger` on multi-rank runs causes output from all ranks
to interleave and overwrite each other.

**Action**: Use `torch.save` to write a separate file per rank. See the Tensor dump
section for the pattern.

---

### KV cache pollution bug appears intermittently

**Symptom**: Same prompt gives different results across runs; the precision issue is
not consistently reproducible.

**Cause**: Warmup leaves stale KV cache entries that corrupt subsequent inference runs.

**Action**: Disable warmup on both lmdeploy+dlinfer and vllm+vllm-ascend, then retry.

---

### Generated tokens differ between runs on the same prompt

**Symptom**: Token-level comparison is not stable; the outputs change each run.

**Cause**: `temperature > 0` introduces sampling randomness that makes comparison
meaningless.

**Action**: Set `temperature=0, top_k=1` on both sides.

---

### Binary search finishes but the bug was in an early layer all along

**Symptom**: After several bisection steps the divergent layer turns out to be very
early in the model.

**Cause**: Starting at layer N/2 skips checking whether layer 0 is already wrong.

**Action**: Always verify layer 0 first. If layer 0 is clean, then apply binary search
to the remaining layers.

---

### Found the suspected divergent op, inputs look identical, root cause unclear

**Symptom**: Inputs to the op match between frameworks, but you cannot tell whether
the op itself is at fault.

**Cause**: Without output dumps, there is no evidence of whether the op produces
wrong results.

**Action**: Dump both inputs and outputs. If inputs match but outputs differ, the bug
is in the op call itself — check argument order, dtype, and shape passed to the hardware op (e.g. NPU op on Ascend).