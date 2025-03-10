import os
import math
import vllm
import torch
import torch.distributed as dist

from vllm import _custom_ops as custom_ops
from flash_attn import flash_attn_varlen_func
from vllm.attention.ops.prefix_prefill import context_attention_fwd

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

from .maca_extension import ops as maca_ext_ops
from .context_flashattention import (
    context_attention_fwd as context_attention_fwd_mla,
)


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


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    custom_ops.fused_add_rms_norm(hidden_states, residual, weight, epsilon)
    return hidden_states, residual


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    position_ids_1d = torch.arange(0, query.size(1), device=query.device)
    head_size = query.size(-1)
    query = query.flatten(-2, -1)
    key = key.flatten(-2, -1)
    rot_dim = cos.size(-1)

    maca_ext_ops.rotary_embedding(
        position_ids_1d,
        query,
        key,
        head_size,
        cos.view(-1, rot_dim),
        sin.view(-1, rot_dim),
        True,
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
    if q_seq_len is None:
        q_seq_len = max_q_seq_len
    kv_seq_len = q_seq_len
    max_kv_seq_len = max_q_seq_len

    causal = True
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(key.size(-1)))

    is_mla = key.size(-1) != value.size(-1)

    if is_mla:
        batch_size = kv_seq_len.size(0)
        head_dim = query.shape[-1]
        nope_size = value.shape[-1]
        groups = num_q_heads // num_q_heads
        value = torch.nn.functional.pad(value, [0, head_dim - nope_size], value=0)

        input_type = query.dtype
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)

        # (bs, seq_len, num_head, head_dim)
        query = query.view(batch_size, -1, num_q_heads, head_dim)
        key = key.view(batch_size, -1, num_kv_heads, head_dim)
        value = value.view(batch_size, -1, num_kv_heads, head_dim)
        key = key.repeat(1, 1, groups, 1)
        value = value.repeat(1, 1, groups, 1)

        # (bs, num_head, seq_len, head_dim)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # (bs, num_head, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=softmax_scale
        )

        # (seq_len, num_head, head_dim)
        attn_output = attn_output.transpose(1, 2).flatten(0, 1)
        attn_output = attn_output[..., :nope_size].to(input_type)
        return attn_output

    # for cogvlm vl part.
    if query.size(-2) != num_q_heads:
        causal = False
        head_dim = query.size(-1) // num_q_heads
        query = query.view(-1, num_q_heads, head_dim)
        key = key.view(-1, num_kv_heads, head_dim)
        value = value.view(-1, num_kv_heads, head_dim)
        q_start_loc = torch.tensor(
            [0, q_seq_len], dtype=torch.int32, device=query.device
        )
        softmax_scale = float(1 / math.sqrt(head_dim))

    output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=q_start_loc,
        cu_seqlens_k=q_start_loc,
        max_seqlen_q=max_q_seq_len,
        max_seqlen_k=max_kv_seq_len,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    return output


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
    maca_ext_ops.reshape_and_cache_new(
        key, value, key_cache, value_cache, kv_indices, "auto", 1.0, 1.0
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
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    num_kv_heads = value_cache.size(1)
    block_size = value_cache.size(-2)
    output = torch.empty_like(query)

    is_mla = query.size(-1) == 576

    if is_mla:
        value_cache = key_cache.transpose(2, 3).reshape(
            -1, num_kv_heads, 576, block_size
        )
    maca_ext_ops.paged_attention_v1(
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
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        1.0,  # k_scale
        1.0,  # v_scale
        torch.cuda.current_device(),  # tp_rank
        0,  # blocksparse_local_blocks
        1,  # blocksparse_vert_stride
        1,  # blocksparse_block_size
        1,  # blocksparse_head_sliding_step
    )
    if is_mla:
        return output[..., :512]
    else:
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

    is_mla = key.size(-1) != value.size(-1)

    if is_mla:
        num_blocks = key_cache.shape[0]
        key_cache = key_cache.permute(0, 1, 2, 4, 3).reshape(
            num_blocks, num_kv_heads, -1, block_size
        )
        value_cache = key_cache
        context_attention_fwd_mla(
            query,
            key,
            value,
            output,
            key_cache,
            value_cache,
            b_loc=block_table,
            b_start_loc=q_start_loc,
            b_seq_len=kv_seq_len,
            b_ctx_len=context_lens,
            max_input_len=max_q_seq_len,
            alibi_slopes=alibi_slopes,
        )
        return output[..., :512]

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

    N = router_logits.size(0)

    topk_weights = torch.empty(
        N, topk, dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(N, topk, dtype=torch.int32, device=router_logits.device)

    token_expert_indicies = torch.empty_like(topk_ids)

    custom_ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        router_logits.float(),
    )

    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)

    return topk_weights, topk_ids


@register_ops(vendor_ops_registry)
def silu_and_mul(x: Tensor, dim: int = -1) -> Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    custom_ops.silu_and_mul(out, x)
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
    N = hidden_states.size(0)
    topk_weights = topk_weights.reshape(N, top_k)
    topk_ids = topk_ids.reshape(N, top_k)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return vllm.model_executor.layers.fused_moe.fused_experts(
        hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids
    )


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
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
    offset = None if (offset is None or offset.numel() == 0) else offset
    output = custom_ops.awq_gemm(x, qweight, scale, offset, group_size)
    if bias is not None:
        output += bias
    return output


@register_ops(vendor_ops_registry)
def dynamic_quant(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = "PER_TOKEN"
):
    assert quant_dtype == torch.int8
    assert quant_granularity == "PER_TOKEN"
    x, input_scale, _ = vllm._custom_ops.scaled_int8_quant(x, None)
    return x, input_scale


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
    assert quant_dtype == torch.int8
    bs, seq_len, head_size = a.size()
    out = vllm._custom_ops.cutlass_scaled_mm(
        a.view(-1, head_size),
        b,
        scale_a=rms_scale,
        scale_b=linear_scale,
        out_dtype=out_dtype,
        bias=bias,
    )
    out = out.view(bs, seq_len, -1)
    return out


@register_ops(vendor_ops_registry)
def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    assert quant_dtype == torch.int8
    x = torch.empty_like(hidden_states)
    vllm._custom_ops.rms_norm(x, hidden_states, weight, epsilon)
    x, input_scale, _ = vllm._custom_ops.scaled_int8_quant(x, None)
    return x, input_scale


@register_ops(vendor_ops_registry)
def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    assert quant_dtype == torch.int8
    vllm._custom_ops.fused_add_rms_norm(hidden_states, residual, weight, epsilon)
    x, input_scale, _ = vllm._custom_ops.scaled_int8_quant(hidden_states, None)
    return x, input_scale, residual
