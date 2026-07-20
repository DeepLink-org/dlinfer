---
name: ascend-nv-timeline-compare
description: Compare inference timelines between Ascend NPU and NVIDIA GPU for the same lmdeploy model/scenario. Part 1 (done) — how to pull an NVIDIA (CUDA backend) decode timeline the same way the ascend-timeline skill pulls an Ascend one, including the CUDA-side audit-marker patch, delay selection on fast GPUs, and phase/batch/KV validation. Part 2 (TODO) — how to actually compare NV vs Ascend performance.
---

# Ascend vs NVIDIA Timeline Comparison

Goal: capture the **same scenario** (model, batch, context, parallelism, decode/prefill
phase) on both Ascend NPU and NVIDIA GPU, then compare. This skill is the NV-side
companion to `ascend-timeline`. Read `ascend-timeline` first — the profiling concepts,
deliverables, and analysis methods there apply verbatim; this skill only records what is
**different on NVIDIA** and how to keep the two runs comparable.

> Status: **Part 1 (pull NV timeline) is written and validated.** Part 2 (NV↔Ascend
> performance comparison) is a TODO stub at the bottom, to be filled in later.

---

## Part 0 — What is the same, what is different

The profiling machinery in lmdeploy is **device-agnostic**. `profiler.py`
(`AgentProfiler`) uses `torch.profiler.profile(activities=[CPU, CUDA])` and
`export_chrome_trace()` on every backend. So all of this is identical to Ascend:

- env vars: `LMDEPLOY_PROFILE_CPU/CUDA/DELAY/DURATION/OUT_PREFIX`
- output naming: `<PREFIX><rank>.json`, Chrome Trace JSON, one file per rank
- delay/duration semantics: `asyncio.sleep(delay)` → `profiler.start()` →
  `asyncio.sleep(duration)` → `dump()`
- phase markers: `forward_eager` (prefill/eager), `forward_cudagraph` (graph decode)
- the fixed-real-batch / KV-context audit **concept**

What differs on NVIDIA:

| Aspect | Ascend | NVIDIA |
|--------|--------|--------|
| Backend | dlinfer vendor ops patched into lmdeploy | lmdeploy **native CUDA backend**, no dlinfer |
| `device_type` in `PytorchEngineConfig` | `"ascend"` | omit / `"cuda"` (default) |
| Profiler routing | `torch_npu.profiler` | stock `torch.profiler` CUDA |
| Audit-marker patch site | `dlinfer/framework/lmdeploy_ext/cudagraph/ascend_cudagraph.py` → `AscendSingleGraphRunner.forward` | `lmdeploy/pytorch/backends/cuda/graph_runner.py` → `CUDASingleGraphRunner.forward` |
| `HCCL_OP_EXPANSION_MODE=AI_CPU` | required | **not used** |
| Collective / compute kernel names in trace | `HcclAllReduce`, `GroupedMatmul`, `FusedInferAttentionScore`, `MatMulV2`, AI CPU kernels | `ncclDevKernel_AllReduce*`, cutlass/`sm90_*`/`ampere_*` gemm, flash-attn kernels, `void at::native::*` |
| Port-conflict cause between runs | Ray holds **HCCL** ports ~45 s | Ray holds GCS/worker ports; still wait between serial tp runs |

Keep everything else (model path, `batch_size`, `input_len`, `output_len`,
`num_speculative_tokens`, `block_size`, `max_batch_size`, `cache_max_entry_count`, target
phase, target real batch, target KV-context band) **identical across the two devices** —
that is the whole point of a comparison run.

---

## Part 1 — Pull an NVIDIA decode timeline

Worked example that produced validated timelines: **metamoe (Qwen3.5 MoE + qwen3_5_mtp,
4 spec tokens), bs=64, KV context 150–250, decode, tp=2 and tp=1, on H200**.

### 1.1 Environment layout (container vs host)

On the reference machine, lmdeploy + the Python env (`/opt/py3`, torch 2.10+cu128) live
**inside a container**; you attach via `tmux attach -t <session>`. `ssh <host>` lands on
the **host**, where that env is absent. But the code tree and model live on a **shared
mount** (`/mnt/...`) visible from both.

Consequences (generalize per machine, don't hard-code paths):

- **Edit code from the host** (shared mount) with normal tools; the container's editable
  lmdeploy install picks it up.
- **Run lmdeploy inside the container** via `tmux send-keys -t <session> "<cmd>" Enter`
  and read progress from a log file on the shared mount (`tmux capture-pane` or
  `tail` the log).
- Put all scripts / logs / timelines in a shared-mount workdir. Watch **ownership**: a
  dir created by container-root may not be writable by the host user — create the workdir
  as the host user (or `chmod 777`) so `scp` from the host works.
- **Never** broadly `kill -9` by grep pattern on a shared host — you can kill other users'
  Ray jobs. Rely on clean process exit + a wait between serial runs.

### 1.2 Audit-marker patch (CUDA single graph runner)

Same purpose as the Ascend patch, different file. Add a **zero-device-sync** marker
emitter to `CUDASingleGraphRunner.forward` in
`lmdeploy/pytorch/backends/cuda/graph_runner.py`. Back up the file first (`.bak`).

Near the top of the module (after `logger = get_logger('lmdeploy')`):

```python
# --- decode fixed-batch / KV-context audit (skill: ascend-nv-timeline-compare) ---
import os as _os

_AUDIT_BS = _os.getenv('LMDEPLOY_AUDIT_DECODE_BATCH_SIZE') == '1'
_AUDIT_KV = _os.getenv('LMDEPLOY_AUDIT_KV_CONTEXT') == '1'


def _emit_decode_audit_markers(context):
    """CPU-side record_function markers for the fixed-real-batch decode audit.

    All values come from CPU scalars: batch = q_seqlens tensor SHAPE (no sync),
    max/sum = StepContext ints. No .item() on a device tensor in the replay
    path, so the measured timeline is not perturbed. For a uniform-length batch
    mean == max, hence min == max.
    """
    if not (_AUDIT_BS or _AUDIT_KV):
        return
    if not getattr(context, 'is_decoding', False):
        return
    q_seqlens = getattr(context, 'q_seqlens', None)
    if q_seqlens is None:
        return
    bs = int(q_seqlens.size(0))  # real scheduler batch, pre graph-padding
    if _AUDIT_BS:
        with record_function(f'decode_actual_batch_size={bs}'):
            pass
    if _AUDIT_KV and bs > 0:
        max_kv = int(getattr(context, 'max_kv_seqlen', 0) or 0)
        sum_kv = int(getattr(context, 'sum_kv_seqlen', 0) or 0)
        mean_kv = sum_kv / bs
        min_kv = max_kv  # no CPU per-seq min scalar; uniform batch -> min==max
        with record_function(
                f'decode_kv_context_min={min_kv},max={max_kv},mean={mean_kv:.1f}'):
            pass
```

Then call it inside `forward`, right after the context is fetched / cudagraph context is
updated and **before** `self._graph.replay()`:

```python
    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        _emit_decode_audit_markers(context)          # <-- inserted
        if self.USE_GRAPH:
            ...
```

Why this is zero-sync and safe (same rule the Ascend skill states — do **not** `.item()`
an NPU/GPU KV tensor in the replay path):

- `q_seqlens.size(0)` is a **tensor shape**, read on CPU, no device sync. It is the real
  scheduler batch **before** graph padding (padding goes into buffers, not into
  `context.q_seqlens`). This is the definition of "real batch".
- `context.max_kv_seqlen` and `context.sum_kv_seqlen` are **CPU Python ints** already
  carried on `StepContext` (computed at input construction in
  `pytorch/engine/inputs_maker.py`, advanced each step). No sync.

Caveats to record in the report (observed with MTP):

- `min` is reported as `= max` because there is no CPU-side per-sequence min scalar
  without threading one through `ModelInputs`/`StepContext`. For a **uniform-length**
  profiling workload (identical prompts, lockstep decode) `min == max` exactly, and this
  is **self-verifying**: if `mean == max` on a marker, that context is uniform.
- The `mean` field can diverge from `max` on **MTP draft-head** replays, because the
  draft-head `StepContext` has different `sum_kv_seqlen`/batch semantics. Treat the
  `min==max` value (per-sequence KV context) as the reliable number; treat `mean` as
  reliable only when it equals `max`.
- If you need an exact per-seq `min` for a **non-uniform** batch, thread a
  `min_kv_seqlen` int through `inputs_maker.py` (next to the existing
  `kv_seqlens.max()/.sum()` — those already run on a CPU tensor, no GPU sync) → `ModelInputs`
  (field + `.step()` + `.clone`) → `StepContext.new()`, then read it here. Higher risk;
  skip it unless the workload is non-uniform.

Verify after patching, from the **container** Python:

```python
import lmdeploy.pytorch.backends.cuda.graph_runner as g
assert hasattr(g, '_emit_decode_audit_markers')
```

### 1.3 Profiling script (CUDA)

Identical to the `ascend-timeline` template except `device_type` is omitted (defaults to
`cuda`) and there is no `HCCL_OP_EXPANSION_MODE`. Must use the `lmdeploy.pipeline` API
(not raw `Engine` + threaded `stream_infer`). Key knobs from the worked example:

```python
backend = PytorchEngineConfig(
    tp=args.tp, cache_max_entry_count=0.6, max_batch_size=64,
    block_size=128, eager_mode=False,          # eager_mode=False => graph decode
)
spec = SpeculativeConfig(method='qwen3_5_mtp', num_speculative_tokens=4)
pipe = lmdeploy.pipeline(model_path, backend_config=backend,
                         speculative_config=spec, trust_remote_code=True)
gen = GenerationConfig(do_sample=False, max_new_tokens=OUT, ignore_eos=True)
```

Add **wall-clock logging** in the harness (`print(f"[WALL {time.time():.3f}] ...")`) at
engine-ready (t0) and per round. You correlate these wall timestamps with the profiler's
`"Profiler start on rank[...]"` log line to confirm the window landed where you intended.

### 1.4 Delay selection on a fast GPU (the crux)

The profiler delay clock starts at `agent.start()` (≈ when `pipeline()` returns, i.e. t0),
**not** at first request. On an H200, MTP decode is **very fast** (~150 KV tokens/s per
sequence), so KV context grows ~150 within any 1 s window — you **cannot** keep a full 1 s
decode window inside a 100-wide context band like [150,250], because
`LMDEPLOY_PROFILE_DURATION` is an **integer ≥ 1 s**.

Strategy that works (keeps every captured replay in-band and makes the window easy to hit):

1. **Short `output_len` so each round's whole decode stays in the band.** With prompt
   ≈ 150 tokens and `output_len=64`, per-sequence context runs 154→~218 across the entire
   round — every decode replay is in [150,250]. (Tokenizer note: this model is
   ~5.36 chars/token; `input_len=166` gave 153 tokens. Confirm from the log's
   `input_tokens=` field, not from the char heuristic.)
2. **Many back-to-back rounds** (`total_rounds=30`) so the decode phase spans ~30 s and
   the window lands in decode regardless of small timing error.
3. **Pick `delay` mid-rounds.** Dry-run once (no profiling env vars) to get the cadence:
   in the example, warmup done ~0.7–2.9 s, rounds ~1.0–1.4 s each. `delay=15` landed the
   1 s window around round 10–14 — solidly in decode, big margin either side.
4. `LMDEPLOY_PROFILE_DURATION=1`.

If the whole 1 s window physically cannot fit the requested band, that is fine: capture
early decode with the band starting at the prompt length, and **report the exact per-replay
context from the markers** rather than claiming a tighter band than the physics allow.

### 1.5 Run env vars

```bash
export LMDEPLOY_PROFILE_CPU=1
export LMDEPLOY_PROFILE_CUDA=1
export LMDEPLOY_PROFILE_DELAY=15          # measured, mid-rounds
export LMDEPLOY_PROFILE_DURATION=1
export LMDEPLOY_PROFILE_OUT_PREFIX=<workdir>/metamoe_tp2_decode_
export LMDEPLOY_AUDIT_DECODE_BATCH_SIZE=1
export LMDEPLOY_AUDIT_KV_CONTEXT=1
# NO HCCL_OP_EXPANSION_MODE on NVIDIA
```

### 1.6 Serial tp runs

Run tp=2 and tp=1 (or any parallel settings you also captured on Ascend) **serially**.
Between runs: let the previous `pipeline` process exit cleanly, confirm GPUs are free
(`nvidia-smi`, 0 MiB), and wait for Ray ports to release. Do **not** broad-kill on a shared
host. Each run reloads the model, so minimize iterations.

### 1.7 Validation (mandatory before trusting a file)

Per rank JSON, grep the markers (works fine on ~80–90 MB files):

```bash
grep -o "forward_cudagraph"            f.json | wc -l   # decode graph replays present?
grep -o "forward_eager"                f.json | wc -l   # prefill events (mixed window OK)
grep -oE "decode_actual_batch_size=[0-9]+" f.json | sort | uniq -c
grep -oE "decode_kv_context_min=[0-9]+"    f.json | grep -oE "[0-9]+" \
  | awk '{if($1>=150&&$1<=250)a++;else b++} END{print "in:"a" out:"b}'
```

What the worked example showed, and what to check:

- **Phase**: `forward_cudagraph` present ⇒ decode-phase window captured (the required
  marker for a decode timeline). `forward_eager` may also appear; a mixed window is valid
  as long as the decode marker is present.
- **Real batch**: `decode_actual_batch_size=64` is the **main-model** decode batch
  (tp2: 128 markers; tp1: 83). A second value `=16` also appears (tp2: 33; tp1: 58) — these
  are the **MTP draft-head** replays that every MTP decode step contains. So an MTP timeline
  is *not* "purely 64"; report both, and state that the main decode is 64.
- **KV context**: min==max on every marker (uniform batch confirmed), range 154–226,
  **all markers in [150,250]** on both ranks.
- **Rank agreement**: both ranks reported identical marker counts.
- **Profiler-stop artifact**: the round active when `dump()` fires is abnormally slow
  (example: round 14 = 2.7 s vs ~1.1 s neighbors). This is the known
  `npu_profile.stop()`/`profiler.stop()` stall — do **not** treat that round as workload
  perf. Same behavior on both devices.

Rigor note: counting markers + rank agreement + in-window presence is a *good* check, but
the **gold-standard** qualification (from `ascend-timeline`) is stronger — verify by
timestamp containment that each `decode_actual_batch_size=B` interval is nested 1:1 inside a
distinct `forward_cudagraph` interval (same pid/tid, start/end containment). Do that pass
when the conclusion must be airtight.

### 1.8 Finding bs=64 in the trace (Perfetto)

1. Open the rank JSON in [ui.perfetto.dev](https://ui.perfetto.dev) (>200 MB → Perfetto;
   <200 MB → chrome://tracing also works).
2. Search `decode_actual_batch_size=64`; jump between hits with Enter. These are CPU-thread
   `record_function` slices on the forward thread.
3. Click one → its **parent** slice is `forward_cudagraph` (confirms graph decode), and its
   **sibling** is `decode_kv_context_min=…,max=…` (the KV context of that replay).
4. Box-select that interval → the device/CUDA-stream kernels underneath are what that
   bs=64 decode replay actually ran.
5. To jump precisely: each marker has a microsecond `ts` in the raw JSON; extract with a
   tiny `json.load` script and enter the `ts` in Perfetto.

### 1.9 Deliverables (same as ascend-timeline)

Preserve raw per-rank JSONs, the stdout/stderr log (used to pick delay and diagnose), and a
report covering: window/params, files analyzed, phase validation, real-batch + KV-context
audit (qualified replay count, exact B, KV min/max/mean or ranges, rank agreement), and any
anomalies. `scp` the raw JSONs + logs back for offline analysis.

---

## Part 2 — Compare NV vs Ascend performance  (TODO — to be added)

<!--
To be filled in by the user. Intended contents:
- Normalizing the two runs so a comparison is fair (identical scenario, same phase window,
  same real batch B and KV band verified by the audit markers on BOTH devices).
- Metric definitions to compare: per-step / per-token decode latency, device-busy vs
  bubble time, NOTIFY_WAIT (update_step_context sync) ratio, top compute-kernel time,
  collective/communication time and overlap.
- Kernel-name mapping across devices (Ascend GroupedMatmul/FusedInferAttentionScore/
  HcclAllReduce  <->  NV cutlass/sm90 gemm / flash-attn / ncclDevKernel_AllReduce) so
  "same work" is compared, not "same name".
- How to align the compute-stream composition percentages between torch_npu and CUDA
  traces, and how to present the comparison (tables / deltas), for both prefill and decode,
  and for MTP (main model vs draft head).
-->
