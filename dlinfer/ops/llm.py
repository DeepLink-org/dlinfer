# Copyright (c) 2024, DeepLink. All rights reserved.
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from dlinfer.utils.graph.custom_op import (
    register_custom_op,
    register_custom_op_default_value,
)


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "fused_attention",
    "fill_contiguous_kvcache",
    "get_cache_len",
    "weight_quant_matmul",
]


@register_custom_op("dlinfer::add_rms_norm", ["hidden_states", "residual"])
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    """
    Add the residual connection and then apply Root Mean Square (RMS) Normalization.

    Args:
        hidden_states (Tensor): The input tensor to be normalized.
        residual (Tensor): The residual tensor to be added to the hidden states.
        weight (Tensor): The weight tensor used for normalization.
        epsilon (float): A small constant added to the root mean square to prevent division by zero.

    Returns:
        Tuple[Tensor, Tensor]:
            - The normalized output tensor.
            - The added result of the residual connection.
    """
    return vendor_ops_registry["add_rms_norm"](hidden_states, residual, weight, epsilon)


@register_custom_op("dlinfer::apply_rotary_pos_emb", ["query", "key"])
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Applies rotary position embeddings to the query and key tensors.

    Rotary position embedding is a method of embedding positional information into
    self-attention computations without increasing the model size.

    Args:
        query (Tensor): The query tensor to apply the rotary position embeddings to.
        key (Tensor): The key tensor to apply the rotary position embeddings to.
        cos (Optional[Tensor]): The cosine component of the rotary position embeddings.
        sin (Optional[Tensor]): The sine component of the rotary position embeddings.
        position_ids (Optional[Tensor]): The position ids used to look up the rotary position embeddings.
        cos_sin_cache (Optional[Tensor]): A cache of pre-computed cosine and sine values.

    Note:
        The parameter groups are mutually exclusive:
        - If `cos` and `sin` are both `None`, then `position_ids` and `cos_sin_cache` must both be Tensor.
        - If `position_ids` and `cos_sin_cache` are both `None`, then `cos` and `sin` must both be Tensor.

    Returns:
        Tuple[Tensor, Tensor]:
            - The query tensor with the rotary position embeddings applied.
            - The key tensor with the rotary position embeddings applied.
    """
    return vendor_ops_registry["apply_rotary_pos_emb"](
        query,
        key,
        cos,
        sin,
        position_ids,
        cos_sin_cache,
    )


@register_custom_op_default_value(
    {
        "softmax_scale": None,
        "alibi_slopes": None,
    }
)
@register_custom_op("dlinfer::prefill_attention", ["attn_output"])
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
    """
    Computes the multi-head attention over the query, key, and value tensors.
    This interface is used for prefilling stage in LLM inference without paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        q_start_loc (Tensor): The start location of each query sequence.
        q_seq_len (Tensor): The length of each query sequence.
        max_q_seq_len (int): The maximum length of any query sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return vendor_ops_registry["prefill_attention"](
        query,
        key,
        value,
        q_start_loc,
        q_seq_len,
        max_q_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        softmax_scale,
        alibi_slopes,
        attn_output,
    )


@register_custom_op("dlinfer::fill_kv_cache", ["key_cache", "value_cache"])
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Fills the key-value cache with the provided key and value tensors.

    Args:
        key (Tensor): The key tensor to be stored in the cache.
        value (Tensor): The value tensor to be stored in the cache.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.
        kv_indices (Tensor): The indices specifying where to store the key and value in the cache.

    Returns:
        Tuple[Tensor, Tensor]:
            - The updated key cache tensor.
            - The updated value cache tensor.
    """
    return vendor_ops_registry["fill_kv_cache"](
        key,
        value,
        key_cache,
        value_cache,
        kv_indices,
    )


@register_custom_op_default_value(
    {
        "softmax_scale": None,
        "alibi_slopes": None,
    }
)
@register_custom_op("dlinfer::paged_decode_attention", ["attn_output"])
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    """
    Computes the multi-head attention over the query, key, and value tensors.
    This interface is used for decoding stage in LLM inference with paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key_cache (Tensor): The cacheed key tensor.
        value_cache (Tensor): The cached value tensor.
        block_table (Tensor): A tensor that maps each position in the query sequence to the corresponding
                              block in the key/value cache.
        block_size (int): The size of each block in the input sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
        max_kv_seq_len (int): The maximum length of any key/value sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return vendor_ops_registry["paged_decode_attention"](
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale,
        alibi_slopes,
        attn_output,
    )


@register_custom_op_default_value(
    {
        "softmax_scale": None,
        "alibi_slopes": None,
    }
)
@register_custom_op("dlinfer::paged_prefill_attention", ["attn_output"])
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
    """
    Computes the multi-head attention over the query, key, and value tensors.
    This interface is used for prefilling stage in LLM inference with paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key_cache (Tensor): The cacheed key tensor.
        value_cache (Tensor): The cached value tensor.
        block_table (Tensor): A tensor that maps each position in the query sequence to the corresponding
                              block in the key/value cache.
        block_size (int): The size of each block in the input sequence.
        q_start_loc (Tensor): The start location of each query sequence.
        q_seq_len (Tensor): The length of each query sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
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
        softmax_scale,
        alibi_slopes,
        attn_output,
    )


@register_custom_op("dlinfer::rms_norm", ["hidden_states"])
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """
    Apply Root Mean Square (RMS) Normalization to the input.

    Args:
        hidden_states (Tensor): The input tensor to be normalized.
        weight (Tensor): The weight tensor used for normalization.
        epsilon (float): A small constant added to the root mean square to prevent division by zero.

    Returns:
        Tensor: The normalized output tensor.
    """
    return vendor_ops_registry["rms_norm"](hidden_states, weight, epsilon)


def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    """
    Given router_logits of experts, it computes the probability distributions of experts
    and then selecting topk values and their corresponding indices.

    Args:
        router_logits (Tensor): The input router logits of probability.
        topk (int): The number of top experts to select.

    Returns:
        Tuple[Tensor, Tensor]:
        - The router weight of selected experts.
        - The index of selected experts.
    """
    return vendor_ops_registry["moe_gating_topk_softmax"](router_logits, topk)


# TODO only for internlm on transformers lib.
# see issue #9 for details
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: Sequence[Optional[Tensor]],
) -> Tensor:
    """
    The navie multi-head attention computation with non varlen input

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return vendor_ops_registry["fused_attention"](
        query_states, key_states, value_states, mask
    )


def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Fills the key-value cache with the provided key and value tensors
    in contiguous way.

    Args:
        key (Tensor): The key tensor to be stored in the cache.
        value (Tensor): The value tensor to be stored in the cache.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.

    Returns:
        Tuple[Tensor, Tensor]:
            - The updated key cache tensor.
            - The updated value cache tensor.
    """
    return vendor_ops_registry["fill_contiguous_kvcache"](
        key_cache, value_cache, key_state, value_state
    )


def get_cache_len(cache: Tensor) -> int:
    """
    Get the seq length of input cache tensor.

    Args:
        cache (Tensor): The input cache tensor.

    Returns:
        int: the required length
    """
    return vendor_ops_registry["get_cache_len"](cache)


@register_custom_op_default_value(
    {
        "offset": None,
        "bias": None,
        "all_reduce": bool,
        "group_size": 0,
    }
)
def weight_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    offset: Optional[Tensor],
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group_size: Optional[int],
) -> Tensor:
    """
    Complete a matrix multiplication computation with quantized scenarios as inputs.

    Args:
        x1 (Tensor): The input tensor.
        x2 (Tensor): The quantized weight tensor.
        scale (Tensor): The antiquant scale tensor of quantized weight.
        offset (Optional[Tensor]): An optional antiquant offset tensor of quantized weight.
        bias (Optional[Tensor]): An optional bias tensor of matrix multiplication.
        all_reduce (Optional[bool]): An optional bool describes whether or not all_reduce is required.
        group_size (Optional[int]): An optional group_size of the quantized weight in the per_group algorithm mode.

    Returns:
        Tensor: The output tensor of the matrix product in the quantisation scenario.
    """
    return vendor_ops_registry["weight_quant_matmul"](
        x1, x2, scale, offset, bias, all_reduce, group_size
    )
