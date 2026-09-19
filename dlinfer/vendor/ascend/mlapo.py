# Copyright (c) 2026, DeepLink. All rights reserved.
"""BF16 static-W8A8 MLA prolog using the prebuilt Ascend operator.

LMDeploy's loader already permutes RoPE channels and stores linear weights
as [out, in]. Do NOT repeat vLLM's RoPE permutation or its initial transpose.
"""
import os
from functools import lru_cache

import torch
import torch_npu

from .moe import _find_vllm_ascend_extension, _prepend_env_path


@lru_cache(maxsize=1)
def ensure_mlapo_op():
    """Register the binary operator without importing any vLLM Python code."""
    try:
        getattr(torch.ops._C_ascend, "mla_preprocess")
        return True
    except AttributeError:
        pass
    library = _find_vllm_ascend_extension()
    if library is None:
        return False
    vendor_path = library.parent / "_cann_ops_custom/vendors/custom_transformer"
    if vendor_path.is_dir():
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)
        vendor_lib_path = vendor_path / "op_api/lib"
        if vendor_lib_path.is_dir():
            _prepend_env_path("LD_LIBRARY_PATH", vendor_lib_path)
    try:
        torch.ops.load_library(str(library))
        getattr(torch.ops._C_ascend, "mla_preprocess")
    except (OSError, RuntimeError, AttributeError):
        return False
    return True


def _pack_weight(weight):
    """Pack [out, in] int8 into [1, in/32, out, 32], then physical NZ."""
    out_dim, in_dim = weight.shape
    # Supported dimensions are aligned; no padding or transpose of the
    # logical matrix is required before this blocked storage conversion.
    blocked = weight.reshape(out_dim // 16, 16, in_dim // 32, 32)
    blocked = blocked.permute(2, 0, 1, 3).reshape(1, in_dim // 32, out_dim, 32)
    return torch_npu.npu_format_cast(blocked.contiguous(), 29)


class MlaPreprocessWeights(torch.nn.Module):
    """Prepacked weights and launch wrapper; cache writes use the current stream."""

    def __init__(self, q_a, kv_a, q_b, q_norm, kv_norm, wuk):
        super().__init__()
        qrank = q_a.weight.shape[0]
        self.hidden_size = q_a.weight.shape[1]
        self.qrank = qrank
        self.heads, self.nope_dim, self.latent_dim = wuk.shape
        self.rope_dim = kv_a.weight.shape[0] - self.latent_dim
        self.register_buffer("wdqkv", _pack_weight(
            torch.cat((kv_a.weight, q_a.weight), dim=0)), persistent=False)
        self.register_buffer("wuq", _pack_weight(q_b.weight), persistent=False)
        tensors = dict(
            descale0=torch.cat((kv_a.deq_scale, q_a.deq_scale)),
            bias0=torch.cat((kv_a.quant_bias, q_a.quant_bias)),
            descale1=q_b.deq_scale.detach(),
            bias1=q_b.quant_bias.detach(),
            gamma1=q_norm.weight.detach().to(torch.bfloat16).contiguous(),
            gamma2=kv_norm.weight.detach().to(torch.bfloat16).contiguous(),
            beta1=(q_norm.bias.detach().to(torch.bfloat16).contiguous()
                   if getattr(q_norm, "bias", None) is not None
                   else torch.zeros(qrank, dtype=torch.bfloat16, device=wuk.device)),
            quant_scale0=q_a.input_scale.detach(),
            quant_offset0=q_a.input_offset.detach(),
            quant_scale1=q_b.input_scale.detach(),
            quant_offset1=q_b.input_offset.detach(),
            wuk=wuk.detach().contiguous(),
            unit=torch.ones(1, dtype=torch.bfloat16, device=wuk.device),
        )
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor, persistent=False)

    @classmethod
    def from_attention(cls, attention):
        """Return None for unsupported weights; all checks precede graph capture."""
        if os.environ.get("DLINFER_ASCEND_ENABLE_MLAPO", "1") == "0":
            return None
        q_a = getattr(attention, "q_a_proj", None)
        kv_a = getattr(attention, "kv_a_proj_with_mqa", None)
        q_b = getattr(attention, "q_b_proj", None)
        attrs = ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset")
        if any(m is None or not all(hasattr(m, a) for a in attrs)
               for m in (q_a, kv_a, q_b)):
            return None
        for m in (q_a, kv_a, q_b):
            if (m.weight.dtype != torch.int8 or m.weight.ndim != 2
                    or m.weight.shape[0] % 16 or m.weight.shape[1] % 32
                    or m.input_scale.dtype != torch.bfloat16
                    or m.input_offset.dtype != torch.int8
                    or m.input_scale.numel() != 1 or m.input_offset.numel() != 1
                    or m.deq_scale.dtype != torch.float32
                    or m.quant_bias.dtype != torch.int32):
                return None
        # The custom kernel fixes both RMSNorm epsilons at 1e-6.
        if (attention.q_a_layernorm.eps != 1e-6
                or attention.kv_a_layernorm.eps != 1e-6
                or getattr(attention.kv_a_layernorm, "bias", None) is not None):
            return None
        wuk = attention.kc.weight
        if (wuk.dtype != torch.bfloat16 or wuk.ndim != 3
                or wuk.shape[-1] != 512 or attention.qk_rope_head_dim != 64
                or wuk.shape[1] % 16 or q_a.weight.shape[0] % 64):
            return None
        if not (torch.equal(q_a.input_scale, kv_a.input_scale)
                and torch.equal(q_a.input_offset, kv_a.input_offset)):
            return None
        if not ensure_mlapo_op():
            return None
        return cls(q_a, kv_a, q_b, attention.q_a_layernorm,
                   attention.kv_a_layernorm, wuk)

    def supports(self, x, cos, sin, rope_cache, nope_cache, metadata):
        tokens = x.shape[1] if x.ndim == 3 else 0
        slots = getattr(metadata, "kv_start_indices", None)
        return (
            getattr(metadata, "is_decoding", False)
            and not getattr(metadata, "is_multi_token_decoding", False)
            and not getattr(metadata, "quant_policy", 0)
            and x.shape[0] == 1 and 0 < tokens <= 1024
            and x.shape[-1] == self.hidden_size
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and cos.shape == sin.shape == (tokens, self.rope_dim)
            and rope_cache.ndim == nope_cache.ndim == 4
            and rope_cache.shape[-2:] == (1, self.rope_dim)
            and nope_cache.shape[-2:] == (1, self.latent_dim)
            and rope_cache.shape[:2] == nope_cache.shape[:2]
            and rope_cache.dtype == nope_cache.dtype == torch.bfloat16
            and rope_cache.is_contiguous() and nope_cache.is_contiguous()
            and slots is not None and slots.numel() >= tokens
        )

    def forward(self, x, cos, sin, rope_cache, nope_cache, slots):
        """Return latent Q, RoPE Q, normalized Q-A; update both caches in place."""
        tokens = x.shape[1]
        q_nope = x.new_empty((tokens, self.heads, self.latent_dim))
        q_rope = x.new_empty((tokens, self.heads, self.rope_dim))
        q_c = x.new_empty((tokens, self.qrank))
        torch.ops._C_ascend.mla_preprocess(
            x[0], self.wdqkv, self.descale0, self.gamma1, self.beta1,
            self.wuq, self.descale1, self.gamma2,
            cos.to(torch.bfloat16).contiguous(), sin.to(torch.bfloat16).contiguous(),
            self.wuk, nope_cache, rope_cache,
            slots.reshape(-1)[:tokens].to(torch.int32).contiguous(),
            quant_scale0=self.quant_scale0, quant_offset0=self.quant_offset0,
            bias0=self.bias0, quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1, bias1=self.bias1,
            ctkv_scale=self.unit, q_nope_scale=self.unit,
            cache_mode="krope_ctkv", quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True, q_out0=q_nope, kv_cache_out0=nope_cache,
            q_out1=q_rope, kv_cache_out1=rope_cache, inner_out=q_c,
        )
        return q_nope, q_rope, q_c
