# Copyright (c) 2024, DeepLink. All rights reserved.
"""Thin pass-through wrappers that expose vllm-ascend's ``_C_ascend`` kernels
through dlinfer's ``vendor_ops_registry``.

Each function here is registered into ``vendor_ops_registry`` (key == function
name) and simply forwards ``*args, **kwargs`` to the corresponding
``torch.ops._C_ascend.<asc_op>`` op.  The ``_C_ascend`` extension is dlopen-ed
by :mod:`dlinfer.vendor.ascend._cascend_loader` at vendor import time.

We deliberately use ``*args, **kwargs`` forwarding rather than re-declaring
every op's signature: the V4 model code (ported from vllm-ascend) already knows
the exact op signatures, and forwarding avoids signature drift if vllm-ascend
evolves.  When a stable, owned interface is desired later, these can be
tightened into typed signatures under a ``dlinfer::`` namespace.
"""
import torch

from dlinfer.utils.registry import register_ops
from dlinfer.vendor import vendor_ops_registry
from dlinfer.vendor.ascend._cascend_loader import load_cascend_kernels

# ensure the _C_ascend extension is loaded before any wrapper is called
load_cascend_kernels()

_C = torch.ops._C_ascend


def _forward(asc_op: str):
    """Build a registered pass-through fn for ``torch.ops._C_ascend.<asc_op>``."""

    def _fn(*args, **kwargs):
        return getattr(_C, asc_op)(*args, **kwargs)

    _fn.__name__ = asc_op
    _fn.__doc__ = f"forward to torch.ops._C_ascend.{asc_op}"
    return _fn


# dlinfer-name -> _C_ascend op name (most are identical; a few are prefixed)
_V4_OP_MAP = {
    # ---- mHC (manifold-constrained hyper-connections) ----
    # vllm-ascend pairs npu_hc_pre_v2 (fusion) with npu_hc_post. The composite
    # npu_hc_pre produces post/comb in a format npu_hc_post does not expect,
    # which makes hc_post drop the attn/mlp signal (post-attn std collapses
    # back to ~embedding -> flat filler logits). lmdeploy flows the hidden as
    # 3D [tok, hc_mult, hidden] (same as vllm), so the v2 op -- which vllm
    # proves works on 3D -- is correct here. The earlier "4D x" rationale no
    # longer applies.
    "hc_pre": "npu_hc_pre_v2",
    "hc_post": "npu_hc_post",
    "add_rms_norm_bias": "npu_add_rms_norm_bias",
    # ---- DSA attention: compressor / indexer / sparse ----
    "compressor": "compressor",
    "compressor_metadata": "compressor_metadata",
    "lightning_indexer": "npu_vllm_quant_lightning_indexer",
    "lightning_indexer_metadata": "npu_vllm_quant_lightning_indexer_metadata",
    # On A2 (Ascend910B2) vllm-ascend's DSA path selects the *kv_quant*
    # sparse-attn variant (npu_kv_quant_sparse_attn_sharedkv) with base kwargs
    # {kv_quant_mode=1 (bf16, no quant), tile_size=64, rope_head_dim=64}.
    # The plain npu_sparse_attn_sharedkv op exists on A2 but silently returns
    # zeros without those kwargs -- so the model's attention contributed ~0
    # and generation was garbage. Match the vllm A2 path: kv_quant op variant
    # for both the kernel and its metadata builder.
    "sparse_attn_sharedkv": "npu_sparse_attn_sharedkv",
    "sparse_attn_sharedkv_metadata": "npu_sparse_attn_sharedkv_metadata",
    "sparse_flash_attention": "npu_sparse_flash_attention",
    "kv_quant_sparse_flash_attention": "npu_kv_quant_sparse_flash_attention",
    "store_kv_block": "store_kv_block",
    "store_kv_block_metadata": "store_kv_block_metadata",
    "reshape_and_cache_bnsd": "npu_reshape_and_cache_bnsd",
    "load_index_kv_cache": "load_index_kv_cache",
    # ---- RoPE / norm-quant fusion ----
    "inplace_partial_rotary_mul": "inplace_partial_rotary_mul",
    "rms_norm_dynamic_quant": "npu_rms_norm_dynamic_quant",
    # ---- MoE routing / grouped matmul ----
    "moe_gating_top_k": "moe_gating_top_k",
    "moe_gating_top_k_hash": "moe_gating_top_k_hash",
    "grouped_matmul_swiglu_quant": "grouped_matmul_swiglu_quant",
    "grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_weight_nz": "grouped_matmul_swiglu_quant_weight_nz",
    "swiglu_group_quant": "npu_swiglu_group_quant",
    "dequant_swiglu_quant": "npu_dequant_swiglu_quant",
    # ---- MLA prolog / transpose matmul ----
    "batch_matmul_transpose": "batch_matmul_transpose",
    "mla_preprocess": "mla_preprocess",
}

# register each forwarder under its dlinfer name (= key in _V4_OP_MAP)
for _dlinfer_name, _asc_name in _V4_OP_MAP.items():
    _fn = _forward(_asc_name)
    _fn.__name__ = _dlinfer_name  # registry key is the function __name__
    register_ops(vendor_ops_registry)(_fn)
    globals()[_dlinfer_name] = _fn

del _forward, _dlinfer_name, _asc_name
