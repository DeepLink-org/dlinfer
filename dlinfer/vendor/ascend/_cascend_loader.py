# Copyright (c) 2024, DeepLink. All rights reserved.
# Loads the prebuilt vllm-ascend _C_ascend kernels (compressor / lightning_indexer
# / sparse_attn_sharedkv / hc_pre / hc_post / grouped_matmul_swiglu_quant / ...)
# so that dlinfer vendor wrappers can call ``torch.ops._C_ascend.*``.
#
# The .so is built by vllm-ascend's CMake (``vllm_ascend.vllm_ascend_C``) and is
# a standard PyTorch native extension: dlopen-ing it runs the ``TORCH_LIBRARY``
# static initializers that register all ops into the ``torch.ops._C_ascend``
# namespace.  ``torch.ops.load_library`` does exactly this without requiring the
# vllm_ascend python package to be importable.
#
# Resolution order:
#   1. env ``DLINFER_CASCEND_SO`` (absolute path to the .so)
#   2. a copy bundled inside this package (``_cascend_kernels/*.so``)
#   3. the well-known path under ``/vllm-workspace``
import os
from pathlib import Path

import torch

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_CANDIDATES = sorted((_PKG_DIR / "_cascend_kernels").glob("*.so"))
_DEFAULT_SO = "/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
# vllm-ascend custom CANN op package: the _C_ascend kernels call aclnn* ops
# that are NOT in the stock libopapi.so -- they live in vllm-ascend's own
# `_cann_ops_custom/vendors/custom_transformer` package (built by
# csrc/build_aclnn.sh).  CANN loads them via ASCEND_CUSTOM_OPP_PATH (dir whose
# `vendors/` subdir holds the package); the op_api/lib also needs to be on
# LD_LIBRARY_PATH.  We auto-discover it next to the .so (or via env).
_DEFAULT_OPP_PARENT = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom"

_loaded = False


def _resolve_so_path() -> str:
    env = os.environ.get("DLINFER_CASCEND_SO")
    if env and os.path.exists(env):
        return env
    if _BUNDLED_CANDIDATES:
        return str(_BUNDLED_CANDIDATES[0])
    if os.path.exists(_DEFAULT_SO):
        return _DEFAULT_SO
    raise FileNotFoundError(
        "dlinfer-ascend cannot locate the _C_ascend kernels .so. "
        "Set DLINFER_CASCEND_SO to its path, or bundle it under "
        "dlinfer/vendor/ascend/_cascend_kernels/."
    )


def _setup_custom_opp_env(so_path: str) -> None:
    """Set ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH so the _C_ascend aclnn
    custom ops resolve.  Idempotent; safe to call before load_library."""
    if os.environ.get("DLINFER_CASCEND_OPP_SET") == "1":
        return
    # discover the custom-op vendor dir: env override > next to .so > default
    opp_parent = os.environ.get("DLINFER_CASCEND_OPP") or _DEFAULT_OPP_PARENT
    vendor = os.path.join(opp_parent, "vendors", "custom_transformer")
    if not os.path.isdir(vendor):
        # fall back to a dir co-located with the .so (bundled layout)
        cand = os.path.join(os.path.dirname(so_path), "_cann_ops_custom",
                            "vendors", "custom_transformer")
        if os.path.isdir(cand):
            vendor = cand
        else:
            # no custom-op package found; rely on whatever the caller set.
            os.environ["DLINFER_CASCEND_OPP_SET"] = "1"
            return
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = \
        vendor + ":" + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    op_api_lib = os.path.join(vendor, "op_api", "lib")
    if os.path.isdir(op_api_lib):
        os.environ["LD_LIBRARY_PATH"] = \
            op_api_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["DLINFER_CASCEND_OPP_SET"] = "1"


def load_cascend_kernels() -> str:
    """dlopen the _C_ascend extension and register its ops. Idempotent."""
    global _loaded
    if _loaded:
        return "already-loaded"
    path = _resolve_so_path()
    _setup_custom_opp_env(path)
    torch.ops.load_library(path)
    _loaded = True
    return path


def is_loaded() -> bool:
    return _loaded


def has_op(op_name: str) -> bool:
    """True if a given _C_ascend op is registered (useful for sanity checks)."""
    return hasattr(torch.ops._C_ascend, op_name)
