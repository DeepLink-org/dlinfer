import os
import math
import torch
import lmdeploy.pytorch.distributed as dist

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

# from vllm._C import ops
# from vllm._C import cache_ops
from vllm import _custom_ops as custom_ops
from vllm.vllm_flash_attn import flash_attn_varlen_func

# from xformers import ops as xops
# from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
#                                          LowerTriangularMaskWithTensorBias)


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fused_moe",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "silu_and_mul",
    "moe_gating_topk_softmax",
    "linear",
    "weight_quant_matmul",
    "dynamic_quant",
    "linear_w8a8",
    "rms_norm_w8a8",
    "add_rms_norm_w8a8",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    custom_ops.fused_add_rms_norm(
        hidden_states,
        residual,
        weight,
        epsilon
    )
    return hidden_states, residual


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    position_ids_1d = torch.arange(0, query.size(0), device=query.device)
    query = query.flatten(-2, -1)
    key = key.flatten(-2, -1)
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2 :]
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    custom_ops.rotary_embedding(
        position_ids_1d, query, key, cos_sin_cache.size(-1), cos_sin_cache, True
    )
    return query, key


@register_ops(vendor_ops_registry)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not support alibi_slopes yet")
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    if q_seq_len is None:
        q_seq_len = max_q_seq_len
    kv_seq_len = q_seq_len
    max_kv_seq_len = max_q_seq_len

    causal = True
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(key.size(-1)))
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=q_start_loc,
        cu_seqlens_k=q_start_loc,
        max_seqlen_q=max_q_seq_len,
        max_seqlen_k=max_kv_seq_len,
        softmax_scale=softmax_scale,
        causal=causal,
        out=attn_output,
    )
    return attn_output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> Tuple[Tensor, Tensor]:
    kv_indices = kv_indices.squeeze(-1)
    kv_scale  = torch.tensor(1., device=key.device)
    custom_ops.reshape_and_cache(
        key, value, key_cache, value_cache, kv_indices, "auto", kv_scale, kv_scale
    )
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not support alibi_slopes yet")

    dim = query.size(-1)
    num_kv_heads = value_cache.size(1)
    block_size = value_cache.size(-1)
    batch_size = block_table.size(0)
    kv_scale  = torch.tensor(1., device=query.device)

    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    block_table = block_table.to(torch.int32)
    kv_seq_len = kv_seq_len.to(torch.int32)

    output = torch.empty_like(query)
    if torch.distributed.is_initialized():
        tp_rank = torch.distributed.get_rank()
    else:
        tp_rank = 0
    custom_ops.paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        softmax_scale,
        block_table,
        kv_seq_len,
        block_size,
        max_kv_seq_len,
        None,
        "auto",
        kv_scale,
        kv_scale,
        tp_rank,
        0,
        0,
        64,
        0,
    )
    return output




@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    cu_seq_lens_kv: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    output = torch.empty_like(query)
    context_lens = kv_seq_len - q_seq_len

    value_cache = value_cache.permute(0, 1, 3, 2)
    context_attention_fwd(
        query,
        key,
        value,
        output,
        "auto",
        key_cache,
        value_cache,
        b_loc=block_table,
        b_start_loc=q_start_loc,
        b_seq_len=kv_seq_len,
        b_ctx_len=context_lens,
        max_input_len=max_q_seq_len,
        alibi_slopes=alibi_slopes,
    )
    return output


@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    weight = weight.to(torch.float32)
    output = torch.empty_like(hidden_states)

    custom_ops.rms_norm(output, hidden_states, weight, epsilon)

    return output.to(input_dtype)


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(
    router_logits: Tensor, topk: int, renormalize: bool = False
) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def silu_and_mul(x: Tensor, dim: int = -1) -> Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    torch.ops._C.silu_and_mul(out, x)
    return out


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    top_k: int,
    renormalize: bool,
) -> Tensor:
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    if os.getenv("DLINER_LINEAR_USE_NN_LAYOUT", "0") == "1":
        out = torch.matmul(x, weight)
        if bias is not None:
            out += bias
    else:
        out = torch.nn.functional.linear(x, weight, bias)
    if all_reduce:
        dist.all_reduce(out)
    return out


# Quantification of W4A16 is currently supported and tested.
@register_ops(vendor_ops_registry)
def weight_quant_matmul(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = False,
    group_size: Optional[int] = 0,
):
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def dynamic_quant(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = "PER_TOKEN"
):
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def linear_w8a8(
    a: Tensor,
    b: Tensor,
    rms_scale: float,
    linear_scale: float,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype = torch.int8,
    bias: Tensor = None,
):
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    raise NotImplementedError("Not implemented on ppu.")


@register_ops(vendor_ops_registry)
def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    raise NotImplementedError("Not implemented on ppu.")