import math
import torch

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
    # "moe_gating_topk_softmax",
]


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
    cos = cos[..., :cos.shape[-1] // 2]
    sin = sin.squeeze(0).squeeze(1)
    sin = sin[..., :sin.shape[-1] // 2]
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    maca_ext_ops.rotary_embedding(
        position_ids_1d,
        query,
        key,
        cos_sin_cache.size(-1),
        cos_sin_cache,
        True
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
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if attn_output is None:
        attn_output = torch.empty_like(query)
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(key.size(-1)))

    def make_cu_seqlens(seqlens):
        cu_seqlens = seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        cu_seqlens = torch.cat([cu_zero, cu_seqlens])
        return cu_seqlens

    cu_seqlens = make_cu_seqlens(q_seq_len).int().to(query.device)

    attn_output = maca_ext_ops.flash_attn_varlen_fwd(
        query,
        key,
        value,
        attn_output,
        cu_seqlens,
        cu_seqlens,
        None,
        alibi_slopes,
        max_q_seq_len,
        max_q_seq_len,
        0.0,
        softmax_scale,
        False,
        True,
        -1,
        -1,
        False,
        None,
    )

    return attn_output[0]


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
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if attn_output is None:
        attn_output = torch.empty_like(query)
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    block_table = block_table.int()
    kv_seq_len = kv_seq_len.int().to(query.device)

    # import pdb; pdb.set_trace()
    maca_ext_ops.paged_attention_v1(
        attn_output,
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
    )
    # import pdb; pdb.set_trace()

    return attn_output


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
    raise NotImplementedError("maca paged_prefill_attention")


@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    output = torch.empty_like(hidden_states)
    maca_ext_ops.rms_norm(output, hidden_states, weight, epsilon)
    return output
