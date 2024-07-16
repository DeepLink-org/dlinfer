import sys
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.type_annotation import Tensor, Optional, List


__all__ = [
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
]


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_full: Optional[Tensor],
    sin_full: Optional[Tensor]
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        query, key, cos, sin, position_ids,
        cos_full, sin_full
    )

def context_attention(
    attn_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: List[Tensor],
    attn_qk_scale: Optional[float]=None, 
    alibi_slopes: Optional[List[float]]=None,
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        attn_output,
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
    )

def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        key, value, key_cache, value_cache, kv_indices,
    )

def paged_decode_attention(
    attn_output: Tensor,
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_qk_scale: Optional[float]=None, 
    alibi_slopes: Optional[List[float]]=None,
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        attn_output,
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
    )

def paged_prefill_attention(
    attn_output: Tensor,
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
    attn_mask: Optional[List[Tensor]]=None,
    attn_qk_scale: Optional[float]=None, 
    alibi_slopes: Optional[List[float]]=None,
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        attn_output,
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
    )

def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float = 1e-6
):
    func_name = sys._getframe().f_code.co_name
    return vendor_ops_registry[func_name](
        hidden_states, weight, epsilon
    )
