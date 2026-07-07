---
name: restful-pressure-benchmark
description: Run RESTful pressure benchmarks with lmdeploy's profile_restful_api.py after starting an lmdeploy API service. Use when Codex needs to choose ShareGPT native-length or random-length workloads, set tokenizer/model/client parameters without hard-coding qwen, compare models or max-batch settings, preserve logs, inspect running batch and KV cache behavior, and report throughput and cache/OOM handling.
---

# RESTful Pressure Benchmark

## Required Setup

Before running any client workload, use the `lmdeploy-service-launch` skill to
start the server and obtain:

- `BASE_URL` or `HOST`/`PORT`
- served `MODEL_NAME`
- `MODEL_PATH`
- `LMDEPLOY_ROOT`, supplied by the user or referenced script
- server log path
- server settings such as backend, device, max batch size, and cache values

Do not assume the model is named `qwen`. The client `--model` must match the
served model id or the user-provided model name. If the launch skill chose
`qwen` for a Qwen model or `interns` for an InternS/InternLM-style model, use
that same value here.

By default, set `TOKENIZER_PATH` equal to `MODEL_PATH`. Override it only when
the user or local benchmark script explicitly gives a separate tokenizer.

Locate the client script from the user-provided lmdeploy checkout. Prefer:

```text
<LMDEPLOY_ROOT>/benchmark/profile_restful_api.py
```

If the file is not there, search under `<LMDEPLOY_ROOT>` and record the actual
path used in the report. Do not hard-code an environment-specific absolute path
in this skill or in reusable instructions.

## Workload Selection

Support both workload families. Choose from the user request:

### ShareGPT Native Length

Use this when the user asks for ShareGPT/default/native lengths. Do not set
fixed random lengths.

```bash
timeout <LIMIT_SECONDS> python3 <PROFILE_RESTFUL_API> \
  --backend lmdeploy \
  --dataset-name sharegpt \
  --dataset-path <SHAREGPT_JSON> \
  --tokenizer <TOKENIZER_PATH> \
  --num-prompts <NUM_PROMPTS> \
  --host <HOST> \
  --port <PORT> \
  --model <MODEL_NAME> \
  --disable-warmup
```

If the user asks to force ShareGPT output length, add
`--sharegpt-output-len <TOKENS>`. Otherwise leave the native output lengths.

### Random Lengths

Use this when the user asks for synthetic/random token lengths, including fixed
length cases such as 8k input / 8k output.

```bash
timeout <LIMIT_SECONDS> python3 <PROFILE_RESTFUL_API> \
  --backend lmdeploy \
  --dataset-name random \
  --dataset-path <SHAREGPT_JSON> \
  --random-input-len <INPUT_LEN> \
  --random-output-len <OUTPUT_LEN> \
  --random-range-ratio <RATIO> \
  --tokenizer <TOKENIZER_PATH> \
  --num-prompts <NUM_PROMPTS> \
  --host <HOST> \
  --port <PORT> \
  --model <MODEL_NAME> \
  --disable-warmup
```

Always pass a local `--dataset-path` for random workloads if the script uses
ShareGPT text as prompt material; otherwise it may try to download a dataset.

Choose `--random-range-ratio` intentionally:

- `1`: fixed lengths exactly equal to `--random-input-len` and
  `--random-output-len`
- `0 < ratio < 1`: random lengths in a bounded range
- `0`: lengths may range down to zero; do not use this for fixed-length tests

Confirm the client log prints the expected aggregate `#Input tokens` and
`#Output tokens` before treating a run as valid.

## Time, Prompt Count, And Fairness

Respect the user's time limit. If none is given, use a conservative timeout and
adjust `NUM_PROMPTS` so each run finishes in the requested window.

For comparisons, keep these aligned unless the user asks otherwise:

- workload family and workload parameters
- `NUM_PROMPTS`
- server max batch size and cache settings
- tokenizer rule
- timeout policy
- client flags such as streaming, warmup, and request rate

If a run is interrupted, times out, or used wrong length parameters, mark it
invalid and rerun rather than mixing it into the comparison table.

## Monitoring And Statistics

Preserve both client and server logs. From the client log, report:

- successful request count
- benchmark duration
- total input tokens
- total generated tokens
- request throughput
- input token throughput
- output token throughput

From the server log, extract `Engine (running/waiting)` and `KV cache` samples.
Report:

- running batch min, max, average
- count of samples near the configured max batch size
- KV cache maximum
- counts of samples at or above 99%, 99.9%, and 100%
- whether there was sustained KV saturation

Do not treat sustained high KV cache as OOM if the server keeps progressing.

## Cache And OOM Policy

Use the cache policy from the user request or service setup. If the user says to
start from `CACHE_MAX_ENTRY_COUNT=0.8`, do so for every run. Reduce cache only
after a confirmed OOM, using the requested decrement step. Record the value used
for every test.

Do not change cache for:

- wrong tokenizer/model name
- dataset/proxy/download failure
- port conflict
- device visibility failure
- ordinary KV cache saturation with ongoing progress

## Failure Handling

- If the server is not ready, return to `lmdeploy-service-launch`.
- If the client cannot find the served model, query `/v1/models` and rerun with
  the served id.
- If token counts do not match the intended workload, fix client parameters and
  rerun.
- If MTP/speculative decoding is requested and fails, stop that series and
  report the exact failing log path and error snippet.
- Always stop the server after the final client run.

## Report Checklist

Write a run report with:

- server setup from `lmdeploy-service-launch`
- lmdeploy root and `profile_restful_api.py` path
- client command and workload type
- model path, served model name, tokenizer path
- dataset path, prompt count, requested lengths, and observed token totals
- timeout and whether the run completed
- throughput table
- speed ratios and percent gaps for comparisons
- running batch and KV cache table
- cache values, OOM status, retries, and failures
- raw log paths

Keep benchmark results run-specific. Do not hard-code previous model numbers in
the skill.
