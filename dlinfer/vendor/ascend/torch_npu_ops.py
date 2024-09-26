# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch
import torch_npu

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
    if len(cos.shape) < 4:
        cos = cos.unsqueeze(2)
    if len(sin.shape) < 4:
        sin = sin.unsqueeze(2)
    query = query.contiguous()
    key = key.contiguous()
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

    if attn_mask is not None:
        seq_qlen_list = None if q_seq_len is None else q_seq_len.cumsum(0).tolist()
        seq_kvlen_list = seq_qlen_list
        scale_value = (
            softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
        )
        attn_output[:] = torch.ops.npu.npu_fusion_attention(
            query,
            key,
            value,
            num_q_heads,
            "TND",
            scale=scale_value,
            atten_mask=attn_mask[0],
            actual_seq_qlen=seq_qlen_list,
            actual_seq_kvlen=seq_kvlen_list,
        )[0]
    else:
        # For now, the value of attn_mask is None only in vit
        seq_len_list = None if q_seq_len is None else q_seq_len.tolist()
        scale_value = 1.0 / math.sqrt(query.shape[-1] // num_q_heads)
        attn_output[:] = torch.ops.npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            actual_seq_lengths=seq_len_list,
            num_heads=num_q_heads,
            scale_value=scale_value,
            input_layout="BSH",
            num_key_value_heads=num_kv_heads,
        )
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
    query = query.view(bs, 1, num_q_heads * dim)
    kv_cache_len = key_cache.shape[0]
    key_cache = key_cache.view(1, kv_cache_len, -1)
    value_cache = value_cache.view(1, kv_cache_len, -1)
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

    # cann incre_fa don't support paged_attn when q_seq_len > 1
    batch = q_start_loc.shape[0]
    q_seq_len_list = q_seq_len.tolist()
    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous()
    for i in range(batch):
        start = q_start_loc[i]
        mask = attn_mask[i]
        for j in range(q_seq_len_list[i]):
            single_q = query[start + j : start + j + 1].view(1, 1, -1)
            single_o = attn_output[start + j : start + j + 1].view(1, 1, -1)
            torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
                single_q,
                key_cache,
                value_cache,
                single_o,
                padding_mask=None,
                atten_mask=mask[j : j + 1],
                actual_seq_lengths=kv_seq_len_list[i : i + 1],
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
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int32
    )
    selected_idx = torch.empty_like(selected_experts)
    return torch.ops.npu_ext.npu_moe_gating_topk_softmax(
        router_logits, None, topk, routing_weights, selected_experts, selected_idx
    )


# TODO only for internlm on transformers lib.
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
                num_q_heads,
                num_kv_heads,
                mask[i : i + 1],
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
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output


@register_ops(vendor_ops_registry)
def weight_quant_matmul(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    group_size: Optional[int] = 0,
):
    offset = None if (offset is None or offset.numel() == 0) else offset
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x, qweight, scale, antiquant_offset=offset, antiquant_group_size=group_size
    )
