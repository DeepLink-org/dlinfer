# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from dlinfer.graph.custom_op import register_custom_op
from dlinfer.vendor import linear_w8a8_scale_type, dynamic_quant_scale_type


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "incre_flash_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "silu_and_mul",
    "moe_gating_topk_softmax",
    "fused_attention",
    "fill_contiguous_kvcache",
    "get_cache_len",
    "weight_quant_matmul",
    "fused_moe",
    "linear",
    "dynamic_quant",
    "linear_w8a8",
    "rms_norm_w8a8",
    "add_rms_norm_w8a8",
    "transdata",
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
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings to the query and key tensors.

    Rotary position embedding is a method of embedding positional information into
    self-attention computations without increasing the model size.

    Args:
        query (Tensor): The query tensor to apply the rotary position embeddings to.
        key (Tensor): The key tensor to apply the rotary position embeddings to.
        cos (Optional[Tensor]): The cosine component of the rotary position embeddings.
        sin (Optional[Tensor]): The sine component of the rotary position embeddings.

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
    )


@register_custom_op(
    "dlinfer::prefill_attention",
    ["attn_output"],
    default_value={
        "softmax_scale": None,
        "alibi_slopes": None,
        "attn_output": None,
    },
)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
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
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.
        q_start_loc (Tensor): The start location of each query sequence.
        q_seq_len (Tensor): The length of each query sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
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


@register_custom_op(
    "dlinfer::incre_flash_attention",
    ["query"],
    default_value={
        "softmax_scale": None,
    },
)
def incre_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    input_layout: str,
    softmax_scale: Optional[float],
) -> Tensor:
    """
    Computes the multi-head attention over the query, key, and value tensors.
    This interface is used for prefilling stage in LLM inference without paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        num_heads (int): The number of query heads.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.

    Returns:
        Tensor: The computed attention output tensor.
    """
    return vendor_ops_registry["incre_flash_attention"](
        query,
        key,
        value,
        num_heads,
        input_layout,
        softmax_scale,
    )


@register_custom_op(
    "dlinfer::fill_kv_cache",
    ["key_cache", "value_cache"],
    default_value={
        "k_scales_zeros": tuple(),
        "v_scales_zeros": tuple(),
        "quant_bits": 0,
    },
)
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
    """
    Fills the key-value cache with the provided key and value tensors.

    Args:
        key (Tensor): The key tensor to be stored in the cache.
        value (Tensor): The value tensor to be stored in the cache.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.
        kv_indices (Tensor): The indices specifying where to store the key and value in the cache.
        k_scales_zeros (Sequence[Optional[Tensor]]): The scales and zeros used to quantify key.
        v_scales_zeros (Sequence[Optional[Tensor]]): The scales and zeros used to quantify value.
        quant_bits (int): The bits which k/v is quantized into.

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
        k_scales_zeros,
        v_scales_zeros,
        quant_bits,
    )


@register_custom_op(
    "dlinfer::paged_decode_attention",
    ["attn_output"],
    default_value={
        "head_size_v": 0,
        "softmax_scale": None,
        "alibi_slopes": None,
        "attn_output": None,
        "kv_scales": None,
        "kv_zeros": None,
        "quant_bits": 0,
    },
)
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
    head_size_v: Optional[int],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
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
        head_size_v (int): The number of value head size.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.
        kv_scales (Optional[Tensor]): The quantization factors for key and value.
        kv_zeros (Optional[Tensor]): The quantization offset for key and value.
        quant_bits (Optional[int]): The bits which k/v is quantized into.

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
        kv_scales,
        kv_zeros,
        quant_bits,
    )


@register_custom_op(
    "dlinfer::paged_prefill_attention",
    ["attn_output"],
    default_value={
        "head_size_v": 0,
        "softmax_scale": None,
        "alibi_slopes": None,
        "attn_output": None,
        "kv_scales": None,
        "kv_zeros": None,
        "quant_bits": 0,
    },
)
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
    head_size_v: Optional[int],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Tensor,
    kv_zeros: Tensor,
    quant_bits: int,
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
        cu_seq_lens_kv (Tensor): The cumulative sequence lengths of the key/value sequences.
        max_q_seq_len (int): The maximum length of any query sequence.
        max_kv_seq_len (int): The maximum length of any key/value sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.
        head_size_v (int): The number of value head size.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.
        kv_scales (Optional[Tensor]): The quantization factors for key and value.
        kv_zeros (Optional[Tensor]): The quantization offset for key and value.
        quant_bits (Optional[int]): The bits which k/v is quantized into.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return vendor_ops_registry["paged_prefill_attention"](
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        block_size,
        q_start_loc,
        q_seq_len,
        kv_seq_len,
        cu_seq_lens_kv,
        max_q_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        softmax_scale,
        alibi_slopes,
        attn_output,
        kv_scales,
        kv_zeros,
        quant_bits,
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


def silu_and_mul_impl_abstract_func(
    input_tensor: Tensor,
    dim_opt: int = -1,
) -> Tensor:
    gate, up = input_tensor.chunk(2, dim_opt)
    assert gate.shape == up.shape
    return gate


@register_custom_op(
    "dlinfer::silu_and_mul",
    default_value={"dim": -1},
    impl_abstract_func=silu_and_mul_impl_abstract_func,
)
def silu_and_mul(
    input_tensor: Tensor,
    dim: int,
) -> Tensor:
    """
    Apply silu activation on the first half part of input tensor along dim, and then do
    elementwise mul between the activated result and second half part of input tensor along dim.

    Args:
        input_tensor (Tensor): The input tensor to be apply silu and mul activation.
        dim (int): The axis that we would split input tensor into two parts.

    Returns:
        Tensor: The activated output tensor.
    """
    return vendor_ops_registry["silu_and_mul"](input_tensor, dim)


def moe_gating_topk_softmax_impl_abstract_func(
    router_logits: Tensor, topk: int
) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int64
    )
    return routing_weights, selected_experts


@register_custom_op(
    "dlinfer::moe_gating_topk_softmax",
    impl_abstract_func=moe_gating_topk_softmax_impl_abstract_func,
)
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


def weight_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = bool,
    group_size: Optional[int] = 0,
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


@register_custom_op("dlinfer::fused_moe", ["hidden_states"])
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
) -> Tensor:
    """
    Implement the Fused Mixture of Experts (MoE) model.

    Args:
        hidden_states (Tensor): The hidden_states tensor.
        top_k (int): The number of top K experts selected among multiple experts.
        topk_ids (Tensor): The IDs of the top K selected experts.
        topk_weights (Tensor): The topk_weights tensor corresponds to the weight of experts in topk_ids.
        gate_up_weights (Tensor): The gate_up_weights tensor used to upsample.
        down_weights (Tensor): The down_weights tensor used to downsample.
        renormalize (bool): A boolean flag to indicate whether to renormalize the output.

    Returns:
        Tensor: The output tensor of the Fused Mixture of Experts (MoE) model.

    """
    return vendor_ops_registry["fused_moe"](
        hidden_states,
        gate_up_weights,
        down_weights,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
    )


def linear_impl_abstract_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    shape_x = x.shape
    shape_w = weight.shape
    rank_w = len(weight.shape)
    assert rank_w in [2, 4], "weight in linear must be a 2D tensor or 4D tensor."
    cx = shape_x[-1]
    cy = shape_w[-1] if rank_w == 2 else shape_w[-1] * shape_w[-3]  # NZ format
    assert (
        cx == cy
    ), f"The last dimension of x must match the last dimension of weight. {cx} != {cy}"
    return x.new_empty((shape_x[:-1] + shape_w[-2:-1]))


@register_custom_op(
    "dlinfer::linear",
    impl_abstract_func=linear_impl_abstract_func,
    default_value={"bias": None, "all_reduce": False, "group": ""},
)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    """
    Complete a linear computation.

    Args:
        x1 (Tensor): The first input tensor of linear computation.
        x2 (Tensor): The second input tensor of linear computation.
        bias (Optional[Tensor]): An optional bias tensor of linear computation.
        all_reduce (Optional[bool]): An optional bool describes whether or not allreduce is required.

    Returns:
        Tensor: The output tensor of linear computation.
    """
    return vendor_ops_registry["linear"](x, weight, bias, all_reduce, group)


def dynamic_quant_impl_abstract_func(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = "PER_TOKEN"
):
    return x.to(quant_dtype), x.new_empty(x.shape[:-1], dtype=torch.float)


@register_custom_op(
    "dlinfer::dynamic_quant",
    impl_abstract_func=dynamic_quant_impl_abstract_func,
    default_value={"quant_granularity": None},
)
def dynamic_quant(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str
) -> Tuple[Tensor, dynamic_quant_scale_type]:
    """
    Perform dynamic quantization on a tensor.

    Args:
        x (Tensor): The input tensor to be quantized.
        quant_dtype (torch.dtype): The data type to which the tensor should be quantized.
        quant_granularity (str, optional): The granularity of quantization. Defaults to "PER_TOKEN".
            Options include:
            - "PER_TOKEN": Quantize each element independently.
            - "PER_CHANNEL": Quantize each channel independently.
            - "PER_TENSOR": Quantize the entire tensor as a whole.

    Returns:
        Tuple[Tensor, dynamic_quant_scale_type]: A tuple containing:
            - The quantized tensor.
            - The scaling factor used during quantization.

    """
    return vendor_ops_registry["dynamic_quant"](x, quant_dtype, quant_granularity)


def linear_w8a8_impl_abstract_func(
    a: Tensor,
    b: Tensor,
    rms_scale: linear_w8a8_scale_type,
    linear_scale: linear_w8a8_scale_type,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias: Tensor,
) -> Tensor:
    res_shape = torch.matmul(a, b.transpose(-1, -2)).shape
    return a.new_empty(res_shape, dtype=out_dtype)


@register_custom_op(
    "dlinfer::linear_w8a8",
    impl_abstract_func=linear_w8a8_impl_abstract_func,
    default_value={"bias": None},
)
def linear_w8a8(
    a: Tensor,
    b: Tensor,
    rms_scale: linear_w8a8_scale_type,
    linear_scale: linear_w8a8_scale_type,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias: Tensor,
) -> Tensor:
    """
    Performs a linear transformation on two quantized input tensors.

    Args:
        a (Tensor): The first quantized input tensor.
        b (Tensor): The second quantized input tensor.
        rms_scale (float): The scaling factor for a.
        linear_scale (float): The scaling factor for b.
        out_dtype (torch.dtype): The target data type for the output tensor.
        quant_dtype (torch.dtype): The data type of the quantized input tensors.
        bias (Tensor): The bias tensor to be added to the output.

    Returns:
        Tensor: The  output tensor after applying the linear transformation.
    """
    return vendor_ops_registry["linear_w8a8"](
        a, b, rms_scale, linear_scale, out_dtype, quant_dtype, bias
    )


def rms_norm_w8a8_impl_abstract_func(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    return hidden_states.to(quant_dtype), hidden_states.new_empty(
        hidden_states.shape[:-1]
    )


@register_custom_op(
    "dlinfer::rms_norm_w8a8",
    impl_abstract_func=rms_norm_w8a8_impl_abstract_func,
)
def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    Apply RMS normalization to the input tensor and quantizes the result.

    Args:
        hidden_states (Tensor): The input tensor to be normalized and quantized.
        weight (Tensor): The scaling weight applied to the normalized tensor.
        epsilon (float): A value added to the denominator for numerical stability during normalization.
        quant_dtype (torch.dtype): The target data type for the quantized result.

     Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - The RMS-normalized and quantized tensor.
            - The scaling factor used during quantization.
    """
    return vendor_ops_registry["rms_norm_w8a8"](
        hidden_states, weight, epsilon, quant_dtype
    )


def add_rms_norm_w8a8_impl_abstract_func(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:
    return (
        hidden_states.to(quant_dtype),
        hidden_states.new_empty(hidden_states.shape[:-1]),
        residual,
    )


@register_custom_op(
    "dlinfer::add_rms_norm_w8a8",
    impl_abstract_func=add_rms_norm_w8a8_impl_abstract_func,
)
def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Apply RMS normalization to the input tensor, adds a residual connection,
    and quantizes the result.

    Args:
        hidden_states (Tensor): The input tensor to be normalized and quantized.
        residual (Tensor): The residual tensor to be added to the normalized tensor.
        weight (Tensor): The scaling weight applied to the normalized tensor.
        epsilon (float): A value added to the denominator for numerical stability during normalization.
        quant_dtype (torch.dtype): The target data type for the quantized result.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - The RMS-normalized, residual-added, and quantized tensor.
            - The scaling factor used during quantization.
            - The residual tensor.
    """
    return vendor_ops_registry["add_rms_norm_w8a8"](
        hidden_states, residual, weight, epsilon, quant_dtype
    )


def transdata_abstract_func(x: Tensor, transdata_type: int):
    assert x.dim() in [2, 3], "x must be 2D or 3D tensor"
    assert transdata_type == 2, "currently transdata_type must be 2"
    assert x.dtype in [
        torch.float16,
        torch.int8,
    ], "x must be float16, int8 tensor"
    bsz = 1 if x.dim() == 2 else x.shape[0]
    m = 16
    n = 16 if x.dtype == torch.float16 else 32
    res = torch.empty(
        size=(bsz, (x.shape[-1] + n - 1) // n, (x.shape[-2] + m - 1) // m * m, n),
        dtype=x.dtype,
        device=x.device,
    )
    return res


@register_custom_op(
    "dlinfer::transdata",
    ["hidden_states"],
    impl_abstract_func=transdata_abstract_func,
    default_value={"transdata_type": 2},
)
def transdata(
    hidden_states: Tensor,
    transdata_type: int,
) -> Tensor:
    """
    NOTE. This is a Ascend specifical function.
    Use ATB TransdataOperation to convert tensor format in graph mode.

    Args:
        hidden_states (Tensor): The input tensor to be transdata format.
        transdata_type (int): If set to 2, means convert from ND to NZ format.

    Returns:
       Tensor : A tensor in target format.
    """
    return vendor_ops_registry["transdata"](hidden_states, transdata_type)
