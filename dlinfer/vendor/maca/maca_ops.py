import math
import torch

from flash_attn import flash_attn_varlen_func
from flash_attn import flash_attn_with_kvcache

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

from .maca_extension import ops as maca_ext_ops

__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "silu_and_mul",
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
    maca_ext_ops.fused_add_rms_norm(hidden_states, residual, weight, epsilon)
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
    query = query.flatten(-2, -1)
    key = key.flatten(-2, -1)
    cos = cos.squeeze(0).squeeze(1)
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin.squeeze(0).squeeze(1)
    sin = sin[..., : sin.shape[-1] // 2]
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    maca_ext_ops.rotary_embedding(
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
    kv_seqlen: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(key.size(-1)))

    # if softmax_scale is not None:
    #     causal = False
    # else:
    causal = True

    def make_cu_seqlens(seqlens):
        if isinstance(seqlens, int):
            cu_seqlens = torch.tensor([0, seqlens], dtype=torch.int32)
            return cu_seqlens
        cu_seqlens = seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        cu_seqlens = torch.cat([cu_zero, cu_seqlens])
        return cu_seqlens

    cu_seqlens_q = make_cu_seqlens(q_seq_len).int().to(query.device)
    cu_seqlens_k = make_cu_seqlens(kv_seqlen).int().to(query.device)
    # causal = False
    # for cogvlm vl part
    if query.size(-2) != num_q_heads:
        causal = False
        head_dim = query.size(-1) // num_q_heads
        query = query.view(-1, num_q_heads, head_dim)
        key = key.view(-1, num_kv_heads, head_dim)
        value = value.view(-1, num_kv_heads, head_dim)
    # import pdb; pdb.set_trace()
    # print(causal, flush=True)
    output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
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
) -> Tuple[Tensor, Tensor]:
    kv_indices = kv_indices.squeeze(-1)

    maca_ext_ops.reshape_and_cache_new(
        key, value, key_cache, value_cache, kv_indices, "auto"
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
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not support alibi_slopes yet")

    batch_size = block_table.size(0)
    _, head, dim = query.shape

    key_cache_t = key_cache.transpose(1, 2)
    value_cache_t = value_cache.transpose(1, 2)

    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    output = flash_attn_with_kvcache(
        query.view(batch_size, -1, head, dim),
        key_cache_t,
        value_cache_t,
        cache_seqlens=kv_seq_len.to(torch.int32).to(query.device),
        block_table=block_table.to(torch.int32),
        softmax_scale=softmax_scale,
        causal=True,
    )
    return output


@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    batch_size = block_table.size(0)
    _, head, dim = query.shape

    key_cache_t = key_cache.transpose(1, 2)
    value_cache_t = value_cache.transpose(1, 2)

    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    output = flash_attn_with_kvcache(
        query.view(batch_size, -1, head, dim),
        key_cache_t,
        value_cache_t,
        cache_seqlens=kv_seq_len.to(torch.int32).to(query.device),
        block_table=block_table.to(torch.int32),
        softmax_scale=softmax_scale,
        causal=True,
    )

    return output


@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    output = torch.empty_like(hidden_states)
    maca_ext_ops.rms_norm(output, hidden_states, weight, epsilon)
    return output


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

    maca_ext_ops.topk_softmax(
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
def silu_and_mul(x: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    maca_ext_ops.silu_and_mul(out, x)
    return out
