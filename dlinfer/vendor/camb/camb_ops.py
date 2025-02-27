import math
import torch
import torch_mlu_ops as tmo

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "fused_moe",
    "linear",
    "dynamic_quant",
    "linear_w8a8",
    "rms_norm_w8a8",
    "add_rms_norm_w8a8",
]


@register_ops(vendor_ops_registry)
def silu_and_mul(input_tensor: Tensor, dim: int) -> Tensor:
    return tmo.active(input_tensor, act_mode="silu", is_gated=True)


@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    dim = hidden_states.ndim
    assert (
        dim == 2 or dim == 3
    ), "only support hidden_states: [total_seq_len, hidden_size] or [bs, seq_len, hidden_size]"
    store_output_before_norm = False
    normed_hidden_states = tmo.fused_rms_norm(
        hidden_states,
        None,
        weight,
        None,
        None,
        epsilon,
        store_output_before_norm,
        None,
        None,
    )
    return normed_hidden_states


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    dim = hidden_states.ndim
    assert (
        dim == 2 or dim == 3
    ), "only support hidden_states: [total_seq_len, hidden_size] or [bs, seq_len, hidden_size]"
    store_output_before_norm = True
    normed_hidden_states, added_hidden_states = tmo.fused_rms_norm(
        hidden_states,
        residual,
        weight,
        None,
        None,
        epsilon,
        store_output_before_norm,
        None,
    )
    return normed_hidden_states, added_hidden_states


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],  # (total_seq_len, head_dim)
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    interleaved = False  # False for fold rope, True for cross rope
    # [1, total_seq_len, q_head_num, head_dim]
    _, total_seq_len, _, head_dim = query.shape

    q_embed = tmo.apply_rotary(
        query,
        sin,
        cos,
        None,
        None,
        interleaved,
        False,
        False,
        total_seq_len,
    )
    k_embed = tmo.apply_rotary(
        key,
        sin,
        cos,
        None,
        None,
        interleaved,
        False,
        False,
        total_seq_len,
    )
    return q_embed, k_embed


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
    assert (
        key.ndim == 3 and value.ndim == 3
    ), "only support key, value: [total_seq_len, head_num, head_size]"
    assert (
        key_cache.ndim == 4 and value_cache.ndim == 4
    ), "only support key_cache, value_cache: [block_num, head_num, block_size, head_size]"
    assert kv_indices.ndim == 1, "only support kv_indices: [total_seq_len]"
    assert kv_indices.dtype == torch.int32, "kv_indices must be torch.int32"

    tmo.reshape_paged_cache(key, value, key_cache, value_cache, kv_indices)

    return key_cache, value_cache


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
    causal = True
    # TODO This is a trick to handle lmdeploy vlmodel vision part flash-attention
    if q_start_loc.shape[0] == q_seq_len.shape[0]:
        q_start_loc = torch.cat([q_start_loc, q_seq_len[-1].unsqueeze(0)])
        causal = True
    if alibi_slopes is not None:
        alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])
    # flash_attention dosen't support in-place operation, so we need to check if attn_output.data_ptr() == query.data_ptr()
    if attn_output is not None and attn_output.data_ptr() == query.data_ptr():
        attn_output = None
    out = tmo.flash_attention(
        query,
        key,
        value,
        attn_output,
        q_start_loc,
        q_start_loc,
        alibi_slopes,
        None,
        max_q_seq_len,
        max_q_seq_len,
        softmax_scale,
        causal,
        -1,
        -1,
        query.dtype,
        False,
    )

    return out


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
    assert query.ndim == 3, "only support q: [batch, head_num ,head_dim]"
    assert (
        key_cache.ndim == 4
    ), "only support k_cache: [num_blocks, kv_head_num, block_size, head_size]"
    assert (
        value_cache.ndim == 4
    ), "only support v_cache: [num_blocks, kv_head_num, block_size, head_size]"
    assert (
        block_table.ndim == 2
    ), "only support bloack_table: [batch_size, max_num_blocks_per_seq]"
    assert block_table.dtype == torch.int32, "only support torch.int32"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    k_cache_quant_scale = None
    v_cache_quant_scale = None
    alibi_slopes = None

    total_seq_len, head_num, head_dim = query.shape
    query_reshaped = query.view(total_seq_len, 1, head_num, head_dim)
    attn_output_reshaped = attn_output.view(total_seq_len, 1, head_num, head_dim)

    tmo.single_query_cached_kv_attn(
        query_reshaped,
        key_cache,
        value_cache,
        attn_output_reshaped,
        block_table,
        kv_seq_len,
        k_cache_quant_scale,
        v_cache_quant_scale,
        alibi_slopes,
        max_kv_seq_len,
        0,
        0,
        softmax_scale,
    )

    return attn_output


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
    if attn_output is not None and attn_output.data_ptr() == query.data_ptr():
        attn_output = None
    output = tmo.flash_attention(
        query,
        key_cache,
        value_cache,
        attn_output,
        q_start_loc,
        cu_seq_lens_kv,
        alibi_slopes,
        None,
        max_seq_len_q=max_q_seq_len,
        max_seq_len_kv=max_kv_seq_len,
        softmax_scale=softmax_scale,
        is_causal=True,
        block_tables=block_table,
    )
    return output


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights, selected_experts = tmo.moe_softmax_topk(router_logits, topk)
    return routing_weights, selected_experts


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
    num_experts = gate_up_weights.shape[0]
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    start_expert_id = 0
    (
        gather_expand_idx,
        gather_combine_idx,
        token_count,
        cusum_token_count,
    ) = tmo.moe_gen_idx(topk_ids, num_experts)
    expand_hidden_states = tmo.moe_expand_input(
        hidden_states,
        gather_expand_idx,
        cusum_token_count,
        start_expert_id,
        num_experts,
    )
    up_proj = tmo.group_gemm(
        expand_hidden_states,
        gate_up_weights,
        token_count,
        expand_idx=None,
        c=None,
        alpha=None,
        beta=None,
    )
    gate_cache = tmo.moe_active(
        up_proj,
        act_mode="silu",
        is_gated=True,
        output=up_proj,
        bias=None,
        cusum_token_count=cusum_token_count,
        start_expert_id=start_expert_id,
        expert_size=num_experts,
    )[:, : (up_proj.size()[-1] // 2)]
    down_proj = tmo.group_gemm(
        gate_cache,
        down_weights,
        token_count,
        expand_idx=None,
        c=None,
        alpha=None,
        beta=None,
    )
    out = tmo.moe_combine_result(
        down_proj,
        reduce_weight=topk_weights,
        gather_ids=gather_combine_idx,
        residual=None,
        cusum_token_count=cusum_token_count,
        start_expert_id=start_expert_id,
        expert_size=num_experts,
        bias=None,
    )
    return out


def _process_input_dim(x: Tensor, scale: Optional[Tensor] = None):
    """
    NOTE: Since torch_mlu_ops matmul kernels requires input to be tow dimension.
    we need to reshape the input tensor to 2D tensor if it is 3D tensor.
    """
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if scale is not None:
        scale = scale.view(-1)
    return x, scale, original_shape


def _process_output_dim(out: Tensor, original_shape: Tuple[int, ...]):
    if original_shape is not None:
        return out.view(*original_shape[:-1], -1)
    return out


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
) -> Tensor:
    x, _, original_shape = _process_input_dim(x, None)
    if all_reduce:
        cncl_comm = torch.distributed.distributed_c10d._world.default_pg._get_backend(
            x.device
        ).get_cncl_comm(x.device.index)
        out = tmo.matmul_allreduce(cncl_comm, x, weight, bias)
    else:
        out = tmo.matmul(x, weight, bias)
    return _process_output_dim(out, original_shape)


@register_ops(vendor_ops_registry)
def dynamic_quant(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = "PER_TOKEN"
):
    assert quant_dtype == torch.int8
    assert quant_granularity == "PER_TOKEN"
    from . import SMOOTH_VEC

    if x.shape[-1] > SMOOTH_VEC.shape[0]:
        from . import update_smooth

        SMOOTH_VEC = update_smooth(x.shape[-1])
    smooth = SMOOTH_VEC[: x.shape[-1]]
    return tmo.per_token_smooth_quantize(x, smooth=smooth)


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
    input_quant, input_scale, original_shape = _process_input_dim(a, rms_scale)
    out = tmo.smooth_quant_matmul(
        input_quant, input_scale, b, linear_scale, out_dtype, bias
    )
    return _process_output_dim(out, original_shape)


@register_ops(vendor_ops_registry)
def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    assert quant_dtype == torch.int8
    store_output_before_norm = False
    normed_hidden_states = tmo.fused_rms_norm(
        hidden_states,
        None,
        weight,
        None,
        None,
        epsilon,
        store_output_before_norm,
        None,
        None,
    )
    x, rms_scale = dynamic_quant(normed_hidden_states, quant_dtype, "PER_TOKEN")
    return x, rms_scale


@register_ops(vendor_ops_registry)
def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
):
    assert quant_dtype == torch.int8
    store_output_before_norm = True
    normed_hidden_states, added_hidden_states = tmo.fused_rms_norm(
        hidden_states,
        residual,
        weight,
        None,
        None,
        epsilon,
        store_output_before_norm,
        None,
    )
    x, rms_scale = dynamic_quant(normed_hidden_states, quant_dtype, "PER_TOKEN")
    return x, rms_scale, added_hidden_states
