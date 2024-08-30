import math
import torch
import torch_mlu
from bangtransformer.torch import bt_ops

from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

__all__ =[
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    # "paged_prefill_attention",
    "rms_norm",
    # "moe_gating_topk_softmax",
    # "get_cache_len",
]

@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float
) -> Tensor:
    store_output_before_norm = False
    dim = hidden_states.ndim
    assert dim == 2 or dim == 3, "only support hidden_states: [total_seq_len, head_size]"
    if dim == 2:
        hidden_states = hidden_states.contiguous()
        normed_hidden_states = bt_ops.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm)[0]
        return normed_hidden_states    
    if dim == 3:
        batch, seqLen, head_size = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = hidden_states.reshape(batch * seqLen, head_size).contiguous()
        normed_hidden_states = bt_ops.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm)[0]
        normed_hidden_states = normed_hidden_states.reshape(batch,seqLen,head_size)
    return normed_hidden_states

@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    assert hidden_states.ndim == 2, "only support hidden_states: [total_seq_len, head_size]"
    store_output_before_norm = True
    normed_hidden_states, added_hidden_states = \
        bt_ops.fused_rms_norm(hidden_states, residual, weight, None, None, epsilon, store_output_before_norm)
    
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
    assert query.ndim == 3, "only support q:[totalSeq, head ,head_dim]"
    assert key.ndim == 3, "only support k:[totalSeq, head ,head_dim]"
    interleaved = False
    embeded_query = torch.empty_like(query)
    embeded_key = torch.empty_like(key)
    if position_ids is not None:
        cos = cos_full[position_ids]
        sin = sin_full[position_ids]
    #view totalSeq as a long sequence
    cu_seq_lens = torch.Tensor([0,query.shape[0]]).long().mlu()
    max_context_len = query.shape[0]
    bt_ops.apply_rotary(embeded_query, query, sin, cos, position_ids, cu_seq_lens, interleaved, True, False, max_context_len)
    bt_ops.apply_rotary(embeded_key, key, sin, cos, position_ids, cu_seq_lens, interleaved, True, False, max_context_len)
    return embeded_query,embeded_key

@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,     
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    assert key.ndim == 3 and value.ndim == 3, \
        "only support key, value: [total_seq_len, head_num, head_size]"
    assert key_cache.ndim == 4 and value_cache.ndim == 4, \
        "only support key_cache, value_cache: [block_num, head_num, block_size, head_size]"
    assert kv_indices.ndim == 1, "only support kv_indices: [total_seq_len]"
    
    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    bt_ops.reshape_paged_cache(key, value, key_cache, value_cache, kv_indices)
    return key_cache, value_cache

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
    max_seq_len = torch.max(seq_len).to(dtype=torch.int32)
    if attn_output == None:
        attn_output = torch.tensor(query.shape())
    if alibi_slopes is not None:
        alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
    softmax_scale = attn_qk_scale
    if attn_qk_scale == None:
        softmax_scale = 1
    total_q_seqLen = int(query.shape[0])
    last = torch.Tensor([total_q_seqLen]).mlu().to(torch.int32)
    cu_seq_len = q_start_loc.to(torch.int32)
    cu_seq_len = torch.cat((cu_seq_len,last),dim=0)
    bt_ops.flash_attention(query, key, value, cu_seq_len, alibi_slopes, None, attn_output, max_seq_len, softmax_scale, True, -1, -1)
    return attn_output

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
    assert query.ndim == 4, "only support q:[batch,seq_q=1, head ,head_dim]"
    assert query.shape[1] == 1, "only support seq_q = 1 in paged decode attention"
    assert key_cache.ndim == 4, "only support k_cache:[num_blocks, kv_head_num, block_size, head_size]"
    assert value_cache.ndim == 4, "only support v_cache:[num_blocks, kv_head_num, block_size, head_size]"
    assert block_table.ndim == 2, "only support bloack_table:[batch_size, max_num_blocks_per_seq]"
    
    batch_size = block_table.shape[0]
    dim = query.shape[3]
    k_cache_quant_scale = None
    v_cache_quant_scale = None
    kv_seq_len = kv_seq_len.to(torch.int32)
    max_context_lens = torch.max(kv_seq_len)

    softmax_scale = 1. / math.sqrt(dim)
    out = attn_output.view_as(query)

    bt_ops.single_query_cached_kv_attn(query, key_cache, value_cache, block_table, kv_seq_len,k_cache_quant_scale, v_cache_quant_scale, alibi_slopes, out, max_context_lens, 0, 0, softmax_scale)

if __name__ == '__main__':
    pass
