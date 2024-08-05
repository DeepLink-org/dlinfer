import math
import torch
import torch_npu

from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

__all__ =[
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
]

@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states= \
        torch.ops.npu.npu_add_rms_norm(hidden_states, residual, weight, epsilon)
    return normed_hidden_states, added_hidden_states

@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_full: Optional[Tensor],
    sin_full: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    if position_ids is not None:
        cos = cos_full[position_ids]
        sin = sin_full[position_ids]
    return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")

@register_ops(vendor_ops_registry)
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
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support alibi_slopes yet")
    if attn_qk_scale is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support attn_qk_scale yet")
    # cann prompt_fa don't support batch query with different seq_len
    seq_len_list = seq_len.tolist()
    if attn_mask:
        batch = q_start_loc.shape[0]
        scale_value = 1. / math.sqrt(query.shape[-1])
        for i in range(batch):
            start = q_start_loc[i]
            end = start + seq_len[i]
            single_seqlen = int(seq_len[i])
            single_q = query[start:end].view(1, single_seqlen, -1)
            single_k = key[start:end].reshape(1, single_seqlen, -1)
            single_v = value[start:end].reshape(1, single_seqlen, -1)
            single_o = attn_output[start:end].view(1, single_seqlen, -1)
            actual_seq_lengths = seq_len_list[i:i+1]
            torch.ops.npu_ext.npu_prompt_flash_attention_out(
                single_q, single_k, single_v, single_o, padding_mask=None,
                atten_mask=attn_mask[i], actual_seq_lengths=actual_seq_lengths, 
                num_heads=num_q_heads, scale_value=scale_value, pre_tokens=2147473647, next_tokens=0,
                input_layout="BSH", num_key_value_heads=num_kv_heads)
    else:
        # For now, the value of attn_mask is None only in vit
        scale_value = 1. / math.sqrt(query.shape[-1] // num_q_heads)
        attn_output[:] = torch.ops.npu.npu_prompt_flash_attention(query, key, value,
            actual_seq_lengths=seq_len_list, num_heads=num_q_heads, scale_value=scale_value,
            input_layout="BSH", num_key_value_heads=num_kv_heads)
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
    key_cache_reshaped = key_cache.view(block_total, head, dim)
    value_cache_reshaped = value_cache.view(block_total, head, dim)
    torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache

@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support alibi_slopes yet")
    if attn_qk_scale is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support attn_qk_scale yet")
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    bs, _, dim = query.shape
    query = query.view(bs, 1, num_q_heads * dim)
    kv_cache_len = key_cache.shape[0]
    key_cache = key_cache.view(1, kv_cache_len, -1)
    value_cache = value_cache.view(1, kv_cache_len, -1)
    scale_value = 1. / math.sqrt(dim)

    torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
        query, key_cache, value_cache, attn_output.view_as(query), padding_mask=None,
        atten_mask=None, actual_seq_lengths=kv_seq_len.tolist(), antiquant_scale=None,
        antiquant_offset=None, block_table=block_table, dequant_scale1=None,
        quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None,
        num_heads=num_q_heads, scale_value=scale_value, input_layout="BSH",
        num_key_value_heads=num_kv_heads, block_size=block_size, inner_precise=1)
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
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support alibi_slopes yet")
    if attn_qk_scale is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support attn_qk_scale yet")
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    # cann incre_fa don't support paged_attn when q_seq_len > 1
    batch = q_start_loc.shape[0]
    q_seq_len_list = q_seq_len.tolist()
    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = 1. / math.sqrt(query.shape[-1])
    for i in range(batch):
        start = q_start_loc[i]
        mask = attn_mask[i]
        for j in range(q_seq_len_list[i]):
            single_q = query[start + j:start + j + 1].view(1, 1, -1)
            single_o = attn_output[start + j:start + j + 1].view(1, 1, -1)
            torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
                single_q, key_cache, value_cache, single_o, padding_mask=None,
                atten_mask=mask[j:j + 1], actual_seq_lengths=kv_seq_len_list[i:i+1],
                antiquant_scale=None, antiquant_offset=None, block_table=block_table,
                dequant_scale1=None, quant_scale1=None, dequant_scale2=None,
                quant_scale2=None, quant_offset2=None,
                num_heads=num_q_heads, scale_value=scale_value, input_layout="BSH",
                num_key_value_heads=num_kv_heads, block_size=block_size, inner_precise=1)
    return attn_output

@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float
) -> Tensor:
    return torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]

@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(
    router_logits: Tensor,
    topk: int
) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty((*router_logits.shape[:-1], topk), dtype=torch.int32)
    selected_idx = torch.empty_like(selected_experts)
    return torch.ops.npu_ext.npu_moe_gating_topk_softmax(router_logits, None, topk, routing_weights,
                                                         selected_experts, selected_idx)
