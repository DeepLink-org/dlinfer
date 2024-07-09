import sys
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.type_annotation import Tensor, Optional, List


def paged_decode_attention(
    attn_output: Tensor,
    query: Tensor,
    cache_key: Tensor,
    cache_value: Tensor,
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
        cache_key,
        cache_value,
        block_table,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_qk_scale, 
        alibi_slopes,
    )
