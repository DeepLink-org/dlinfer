import torch
import torch_npu

# from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, List

# @register_ops(vendor_ops_registry)
# def paged_decode_attention(
#     attn_output: Tensor,
#     query: Tensor,
#     cache_key: Tensor,
#     cache_value: Tensor,
#     block_table: Tensor,
#     block_size: int,
#     kv_seq_len: Tensor,
#     num_q_heads: int,
#     num_kv_heads: int,
#     attn_qk_scale: Optional[float], 
#     alibi_slopes: Optional[List[float]],
# ):
#     if alibi_slopes is not None:
#         raise RuntimeError("paged_decode_attention does not "
#                            "support alibi_slopes yet")
#     if attn_qk_scale is not None:
#         raise RuntimeError("paged_decode_attention does not "
#                            "support attn_qk_scale yet")
#     out = torch.ops.npu.npu_incre_flash_attention(
#         query, cache_key, cache_value, padding_mask=None,
#         attn_mask=None, actual_seq_lengths=kv_seq_len.tolist(), antiquant_scale=None,
#         antiquant_offset=None, block_table=block_table, dequant_scale1=None,
#         quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None,
#         num_heads=num_q_heads, scale_value=1.0, input_layout="BSH",
#         num_key_value_heads=num_kv_heads, block_size=block_size, inner_precise=1)
#     attn_output.copy_(out)
