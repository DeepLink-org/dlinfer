# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch

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
    "get_cache_len",
    "weight_quant_matmul",
    "fused_moe",
    "linear",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    return normed_hidden_states, added_hidden_states


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    # rotary pos emb helpers:
    def rotate_half_(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_(q, k, cos, sin):
        return (q * cos) + (rotate_half_(q) * sin), (k * cos) + (rotate_half_(k) * sin)

    if len(cos.shape) < 4:
        cos = cos.unsqueeze(2)
    if len(sin.shape) < 4:
        sin = sin.unsqueeze(2)
    query = query.contiguous()
    key = key.contiguous()

    # ascend ops currently only support dim 128
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        return apply_rotary_pos_emb_(query, key, cos, sin)
    return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")


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

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    seq_qlen_list = (
        [max_q_seq_len * (i + 1) for i in range(query.shape[0])]
        if q_seq_len is None
        else q_seq_len.cumsum(0).tolist()
    )
    seq_kvlen_list = seq_qlen_list
    if len(attn_mask) == 0 and q_seq_len is None:
        query = query.view(query.shape[0] * query.shape[1], num_q_heads, -1)
        key = key.view(key.shape[0] * key.shape[1], num_kv_heads, -1)
        value = value.view(value.shape[0] * value.shape[1], num_kv_heads, -1)
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    attn_mask_ = None if len(attn_mask) == 0 else attn_mask[0]
    attn_output.view(query.shape)[:] = torch.ops.npu.npu_fusion_attention(
        query,
        key,
        value,
        num_q_heads,
        "TND",
        scale=scale_value,
        atten_mask=attn_mask_,
        actual_seq_qlen=seq_qlen_list,
        actual_seq_kvlen=seq_kvlen_list,
    )[0]
    return attn_output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    head, dim = key.shape[1:]
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()
    kv_indices = kv_indices.view(-1, 1)

    key_cache_reshaped = key_cache.view(block_total, head, dim)
    value_cache_reshaped = value_cache.view(block_total, head, dim)
    torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    key_cache = torch.cat([key_cache, key_state], dim=1)
    value_cache = torch.cat([value_cache, value_state], dim=1)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def get_cache_len(cache: Tensor):
    return cache.shape[1]


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
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    bs, _, dim = query.shape
    query = query.contiguous()
    attn_output = attn_output.contiguous()
    query = query.view(bs, 1, num_q_heads * dim)
    scale_value = 1.0 / math.sqrt(dim)

    torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
        query,
        key_cache,
        value_cache,
        attn_output.view_as(query),
        padding_mask=None,
        atten_mask=None,
        actual_seq_lengths=kv_seq_len.tolist(),
        antiquant_scale=None,
        antiquant_offset=None,
        block_table=block_table,
        dequant_scale1=None,
        quant_scale1=None,
        dequant_scale2=None,
        quant_scale2=None,
        quant_offset2=None,
        num_heads=num_q_heads,
        scale_value=scale_value,
        input_layout="BSH",
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        inner_precise=1,
    )
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
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if softmax_scale is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support softmax_scale yet"
        )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous().view(query.shape[0], 1, -1)
    torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
        query,
        key_cache,
        value_cache,
        attn_output,
        padding_mask=None,
        atten_mask=attn_mask[0],
        actual_seq_lengths=kv_seq_len_list,
        antiquant_scale=None,
        antiquant_offset=None,
        block_table=block_table,
        dequant_scale1=None,
        quant_scale1=None,
        dequant_scale2=None,
        quant_scale2=None,
        quant_offset2=None,
        num_heads=num_q_heads,
        scale_value=scale_value,
        input_layout="BSH",
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        inner_precise=1,
    )
    return attn_output


@register_ops(vendor_ops_registry)
def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    hidden_states = hidden_states.contiguous()
    return torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


@register_ops(vendor_ops_registry)
def silu_and_mul(input_tensor: Tensor, dim: int) -> Tensor:
    return torch.ops.npu.npu_swiglu(input_tensor, dim)


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int32
    )
    selected_idx = torch.empty_like(selected_experts)
    routing_weights, selected_idx = torch.ops.npu_ext.npu_moe_gating_topk_softmax(
        router_logits, None, topk, routing_weights, selected_experts, selected_idx
    )
    return routing_weights, selected_idx.to(torch.int64)


# TODO only for internlm in transformers lib.
# see issue #9 for details
@register_ops(vendor_ops_registry)
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: Sequence[Optional[Tensor]],
) -> Tensor:
    batch_size = query_states.shape[0]
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    q_seq_len, num_q_heads, _ = query_states.shape
    kv_seq_len, num_kv_heads, _ = value_states.shape
    attn_output = torch.empty_like(query_states)

    for i in range(batch_size):
        if q_seq_len == kv_seq_len:
            # mask must be a square
            if not mask[i : i + 1][0].shape[-1] == mask[i : i + 1][0].shape[-2]:
                min_shape = min(
                    mask[i : i + 1][0].shape[-1], mask[i : i + 1][0].shape[-2]
                )
                square_mask = mask[i : i + 1][0][..., :min_shape, :min_shape]
                square_mask = square_mask.contiguous()
            else:
                square_mask = mask[i : i + 1][0]

            prefill_attention(
                query_states,
                key_states,
                value_states,
                torch.tensor(
                    [kv_seq_len - q_seq_len],
                    dtype=torch.int64,
                    device=query_states.device,
                ),
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                q_seq_len,
                num_q_heads,
                num_kv_heads,
                [
                    square_mask,
                ],
                None,
                None,
                attn_output,
            )
        else:
            paged_decode_attention(
                query_states,
                key_states,
                value_states,
                None,
                0,
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                kv_seq_len,
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output


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
) -> Tensor:
    offset = None if (offset is None or offset.numel() == 0) else offset
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x, qweight, scale, antiquant_offset=offset, antiquant_group_size=group_size
    )


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    top_k: int,
    topk_ids: Tensor,
    topk_weights: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
) -> Tensor:
    seq_length = hidden_states.size(0)
    moe_output = torch.zeros_like(hidden_states)

    for i in range(seq_length):
        current_hidden_state = hidden_states[i]

        # faster than remove the for loop
        for j in range(top_k):
            expert_id = topk_ids[i][j]
            weight = topk_weights[i][j]

            up_weight = gate_up_weights[expert_id]
            up_proj = torch.matmul(up_weight, current_hidden_state)

            gate_cache, up_cache = up_proj.chunk(2, -1)
            gate_cache = torch.nn.functional.silu(gate_cache, inplace=True) * up_cache

            down_weight = down_weights[expert_id]
            down_proj = torch.matmul(down_weight, gate_cache)

            moe_output[i] += weight * down_proj
    return moe_output


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
) -> Tensor:
    if all_reduce:
        hcomm_info = torch.distributed.distributed_c10d._world.default_pg._get_backend(
            x.device
        ).get_hccl_comm_name(x.device.index)
        out = torch.ops.npu.npu_mm_all_reduce_base(
            x, weight.transpose(0, 1), hcomm_info, reduce_op="sum", bias=bias
        )
    else:
        out = torch.nn.functional.linear(x, weight, bias)
    return out
