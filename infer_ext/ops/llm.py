from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from infer_ext.utils.graph.custom_op import \
    register_custom_op, register_custom_op_default_value


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "fused_attention",
    "fill_contiguous_kvcache",
    "get_cache_len",
]


@register_custom_op("infer_ext::add_rms_norm",
                    ["hidden_states", "residual"])
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    return vendor_ops_registry["add_rms_norm"](
        hidden_states, residual, weight, epsilon
    )

@register_custom_op("infer_ext::apply_rotary_pos_emb",
                    ["query", "key"])
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_full: Optional[Tensor],
    sin_full: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    return vendor_ops_registry["apply_rotary_pos_emb"](
        query, key, cos, sin, position_ids,
        cos_full, sin_full
    )

@register_custom_op_default_value({
    'attn_qk_scale': None,
    'alibi_slopes': None,
})
@register_custom_op("infer_ext::context_attention", ["attn_output"])
def context_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    return vendor_ops_registry["context_attention"](
        query,
        key,
        value,
        q_start_loc,
        seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        attn_qk_scale,
        alibi_slopes,
        attn_output,
    )

@register_custom_op("infer_ext::fill_kv_cache",
                    ["key_cache", "value_cache"])
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    return vendor_ops_registry["fill_kv_cache"](
        key, value, key_cache, value_cache, kv_indices,
    )

@register_custom_op_default_value({
    'attn_qk_scale': None,
    'alibi_slopes': None,
})
@register_custom_op("infer_ext::paged_decode_attention", ["attn_output"])
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    return vendor_ops_registry["paged_decode_attention"](
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_qk_scale,
        alibi_slopes,
        attn_output,
    )

@register_custom_op_default_value({
    'attn_qk_scale': None,
    'alibi_slopes': None,
})
@register_custom_op("infer_ext::paged_prefill_attention", ["attn_output"])
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
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    return vendor_ops_registry["paged_prefill_attention"](
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        q_start_loc,
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        attn_qk_scale,
        alibi_slopes,
        attn_output,
    )

@register_custom_op("infer_ext::rms_norm", ["hidden_states"])
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    return vendor_ops_registry["rms_norm"](
        hidden_states, weight, epsilon
    )

def moe_gating_topk_softmax(
    router_logits: Tensor,
    topk: int
) -> Tuple[Tensor, Tensor]:
    return vendor_ops_registry["moe_gating_topk_softmax"](
        router_logits, topk
    )


# TODO only for internlm on transformers lib.
# see issue #9 for details
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: list,
) -> Tensor:
    return vendor_ops_registry["fused_attention"](
        query_states, key_states,
        value_states, mask
    )

def fill_contiguous_kvcache(
    key_cache: Tensor,
    value_cache: Tensor,
    key_state: Tensor,
    value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    return vendor_ops_registry["fill_contiguous_kvcache"](
        key_cache, value_cache,
        key_state, value_state
    )

def get_cache_len(cache: Tensor) -> int:
    return vendor_ops_registry["get_cache_len"](cache)

