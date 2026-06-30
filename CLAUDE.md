Please read and strictly follow all specifications and architectural instructions in the AGENTS.md file in the root of this project.

# dlinfer

dlinfer is a hardware abstraction layer that bridges [lmdeploy](https://github.com/InternLM/lmdeploy)'s PyTorch backend to domestic AI accelerators: **Ascend NPU**, **Cambricon MLU (CAMB)**, and **Moore Threads GPU (MACA)**. It provides vendor ops, kernel wrappers, and framework patches so lmdeploy models run on these backends without modifications to the core model code.

## Repository layout

```
dlinfer/
  ops/llm.py               # custom op registry (registers with torch dispatch)
  vendor/ascend/           # Ascend NPU ops (torch_npu_ops.py, attention.py, …)
  vendor/camb/             # Cambricon ops (camb_ops.py)
  vendor/maca/             # Moore Threads ops (maca_ops.py)
  framework/lmdeploy_ext/
    cudagraph/             # graph-mode buffer management per vendor
    device/                # device-level lmdeploy patches (ascend.py, camb.py, …)
    quants/                # quantization patches (AWQ, etc.)
```

lmdeploy lives in a sibling directory and is patched at import time via `framework/lmdeploy_ext/`.

## Commands

### Install dlinfer (source)

```bash
# Ascend
DEVICE=ascend python3 setup.py develop

# MACA
DEVICE=maca python3 setup.py develop

# Cambricon
DEVICE=camb python3 setup.py develop
```

### Install lmdeploy (from source, sibling directory)

```bash
# Ascend
LMDEPLOY_TARGET_DEVICE=ascend pip3 install -e .
# MACA
LMDEPLOY_TARGET_DEVICE=maca   pip3 install -e .
# Cambricon
LMDEPLOY_TARGET_DEVICE=camb   pip3 install -e .
```

### Dev commands

```bash
pytest tests/         # run tests
bash run_format.sh    # format code
```
