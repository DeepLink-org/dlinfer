---
name: lmdeploy-service-launch
description: Start, monitor, and stop lmdeploy REST API services for any supported device/backend. Use when Codex needs to run `lmdeploy serve api_server` for a model path, choose model names from the model being served, set backend/device/max-batch/cache and user-provided parallel strategy flags, run outside the sandbox when devices are unavailable inside it, preserve server logs, check readiness, handle port conflicts, and cleanly stop the service before another run.
---

# LMDeploy Service Launch

## Purpose

Use this skill only for the server side of a RESTful lmdeploy run. Keep it
generic: do not assume Ascend, CUDA, qwen, a fixed model name, or a fixed model
path unless the user or referenced script specifies them.

## Inputs To Resolve

Before launching, resolve these from the user request, referenced scripts, or
local context:

- `MODEL_PATH`: required.
- `MODEL_NAME`: use the user-provided name if present. Otherwise choose a short
  name that matches the model being served, such as `qwen` for Qwen-family
  models or `interns` for InternS/InternLM-family models. If unsure, derive a
  stable lowercase name from the model path basename, then query `/v1/models`
  after startup and use the served id for clients.
- `BACKEND`: default to the referenced script or existing project convention.
- `DEVICE`: use the user-provided device if present. If the user did not specify
  one, infer it from the host with the device-selection checks below. Do not
  hard-code a device in this skill.
- `PORT`: default to the referenced script or an available local port.
- `MAX_BATCH_SIZE`, `CACHE_BLOCK_SEQ_LEN`, `CACHE_MAX_ENTRY_COUNT`, and other
  server capacity flags from the user's comparison setup.
- `PARALLEL_FLAGS`: user-provided parallel strategy flags. Do not assume this is
  `--tp`; it may be tensor parallel, expert parallel, data parallel, or another
  lmdeploy-supported strategy.
- `LOG_FILE`: always write server stdout/stderr to a preserved log file.

If the user asks to compare models, keep all comparable server parameters
aligned unless the user explicitly asks to vary one.

## Device Selection

Respect explicit user input first. If the user specifies `ascend`, `cuda`,
`maca`, `camb`, CPU, or another lmdeploy-supported device, use that value and do
not auto-detect a different one.

If the user does not specify a device, probe the host before launching:

```bash
command -v npu-smi >/dev/null 2>&1 && npu-smi info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi
```

Use the probe result to choose the lmdeploy `--device` value:

- `npu-smi info` succeeds and shows available devices: use `ascend`.
- `nvidia-smi` succeeds and shows available GPUs: use `cuda`.
- both succeed: prefer the device implied by the referenced script, current
  environment variables, or user context; otherwise ask before choosing.
- neither succeeds: check the referenced script or local docs. If still
  unknown, ask the user for `DEVICE` instead of guessing.

Record the probe command and outcome in the run log/report when auto-detection
is used.

## Launch Template

Adapt the command to the current model and device. The template intentionally
uses placeholders instead of qwen- or Ascend-specific defaults:

```bash
cd <WORKDIR>

export PYTHONPATH=<LMDEPLOY_ROOT>:<PROJECT_ROOT>:${PYTHONPATH:-}
# Export device/vendor environment variables only when required by the selected
# stack or referenced script.

lmdeploy serve api_server <MODEL_PATH> \
  --model-name <MODEL_NAME> \
  --backend <BACKEND> \
  --device <DEVICE> \
  --server-port <PORT> \
  --max-batch-size <MAX_BATCH_SIZE> \
  --cache-block-seq-len <CACHE_BLOCK_SEQ_LEN> \
  --cache-max-entry-count <CACHE_MAX_ENTRY_COUNT> \
  <PARALLEL_FLAGS> \
  <EXTRA_SERVER_FLAGS>
```

Omit flags that the local lmdeploy version does not support. If a referenced
script uses a different option name for port or device, follow the script and
local `lmdeploy serve api_server --help`.

## Device And Sandbox Handling

If the server needs accelerator devices and the sandbox cannot see them, run the
service command outside the sandbox with the required approval. Do not try to
measure performance from an environment that cannot enumerate the requested
device.

Treat device visibility failures separately from OOM:

- device not visible: stop and report the environment issue
- port already bound: stop the old service or choose another port
- OOM: apply the benchmark's cache/memory fallback policy if one was given

## Readiness And Health

Poll the server log until one of these happens:

- ready: the log contains `Uvicorn running`
- failure: the process exits or the log contains a traceback/fatal runtime error
- timeout: startup exceeds the expected model load window

After readiness, optionally confirm the served model id:

```bash
curl -s http://127.0.0.1:<PORT>/v1/models
```

Use that id as the client `--model` value unless the user explicitly requires a
different model name.

## Stop And Cleanup

After each benchmark case:

1. Stop the client first.
2. Stop the server cleanly with interrupt/terminate.
3. Wait for the service process to exit.
4. Confirm the port is free before starting another server.
5. Wait for device runtime cleanup if the backend leaves worker processes or
   communication ports alive briefly.

Never leave a service session running at the end of the task.

## Server Report Items

Record these in the run report:

- exact server command or script path
- model path and served model name
- backend, device, port, and parallel strategy flags
- max batch size, cache block length, cache max entry count
- extra flags, including speculative/MTP flags if used
- startup result and log path
- any retries, cache changes, OOMs, port conflicts, or device errors
