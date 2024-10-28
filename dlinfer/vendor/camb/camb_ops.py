import math
import torch
import torch_mlu
import torch_mlu_ops as tmo

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

__all__ =[
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    # "paged_prefill_attention",
    "rms_norm",
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
    assert dim == 2 or dim == 3, "only support hidden_states: [total_seq_len, hidden_size] or [bs, seq_len, hidden_size]"
    store_output_before_norm = False
    if dim == 2:
        normed_hidden_states = tmo.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm, None, None)
        return normed_hidden_states    
    else:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        normed_hidden_states = tmo.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm, None, None)
        normed_hidden_states = normed_hidden_states.view(original_shape)
        return normed_hidden_states

@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    dim = hidden_states.ndim
    assert dim == 2 or dim == 3, "only support hidden_states: [total_seq_len, hidden_size] or [bs, seq_len, hidden_size]"
    store_output_before_norm = True
    if dim == 2:
        normed_hidden_states, added_hidden_states = \
            tmo.fused_rms_norm(hidden_states, residual, weight, None, None, epsilon, store_output_before_norm, None)
        return normed_hidden_states, added_hidden_states
    else:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        residual = residual.view(-1, original_shape[-1])
        normed_hidden_states, added_hidden_states = \
            tmo.fused_rms_norm(hidden_states, residual, weight, None, None, epsilon, store_output_before_norm, None)
        normed_hidden_states = normed_hidden_states.view(original_shape)
        added_hidden_states = added_hidden_states.view(original_shape)
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
    assert query.ndim == 3, "only support q:[totalSeq, head ,head_dim]"
    assert key.ndim == 3, "only support k:[totalSeq, head ,head_dim]"
    interleaved = False
    max_context_len = query.shape[0]
  
    total_seq_len, q_head_num, head_dim = query.shape
    k_head_num = key.shape[1]

    query = query.reshape(total_seq_len, 1, q_head_num, head_dim)
    key = key.reshape(total_seq_len, 1, k_head_num, head_dim)
    sin = sin.reshape(total_seq_len, 1, head_dim)
    cos = cos.reshape(total_seq_len, 1, head_dim)  
    
    query = tmo.apply_rotary(query, sin, cos, None, None, interleaved, False, True, 1)
    key = tmo.apply_rotary(key, sin, cos, None, None, interleaved, False, True, 1)

    query = query.view(total_seq_len, q_head_num, head_dim)
    key = key.view(total_seq_len, k_head_num, head_dim)

    return query, key

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
    assert kv_indices.dtype == torch.int32, "kv_indices must be torch.int32"

    tmo.reshape_paged_cache(key, value, key_cache, value_cache, kv_indices)

    return key_cache, value_cache

@register_ops(vendor_ops_registry)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor, # cu_seqlens
    q_seq_len: Tensor,
    max_q_seq_len: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
    if softmax_scale is None:
        softmax_scale = 1. / math.sqrt(query.shape[-1]) 

    tmo.flash_attention(query, key, value, attn_output, q_start_loc, q_start_loc, alibi_slopes, 
                        None, max_q_seq_len, max_q_seq_len, softmax_scale, True, -1, -1, query.dtype, False)

    return attn_output

@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    assert query.ndim == 3, "only support q: [batch, head_num ,head_dim]"
    assert key_cache.ndim == 4, "only support k_cache: [num_blocks, kv_head_num, block_size, head_size]"
    assert value_cache.ndim == 4, "only support v_cache: [num_blocks, kv_head_num, block_size, head_size]"
    assert block_table.ndim == 2, "only support bloack_table: [batch_size, max_num_blocks_per_seq]"
    assert block_table.dtype == torch.int32, "only support torch.int32"

    if softmax_scale is None:
        softmax_scale = 1. / math.sqrt(query.shape[-1])

    k_cache_quant_scale = None
    v_cache_quant_scale = None
    alibi_slopes = None

    total_seq_len, head_num, head_dim = query.shape
    query = query.reshape(total_seq_len, 1, head_num, head_dim)
    attn_output = attn_output.reshape(total_seq_len, 1, head_num, head_dim)

    tmo.single_query_cached_kv_attn(query, key_cache, value_cache, attn_output, block_table, kv_seq_len, k_cache_quant_scale, v_cache_quant_scale, \
        alibi_slopes, max_kv_seq_len, 0, 0, softmax_scale)
    
    attn_output = attn_output.reshape(total_seq_len, head_num, head_dim)
    return attn_output
