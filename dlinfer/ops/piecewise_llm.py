"""
Piecewise专用算子 - 最简化版
使用 *args/**kwargs 完全透明地转发所有参数给 vendor ops
"""
import torch
from dlinfer.vendor.ascend.torch_npu_ops import vendor_ops_registry
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.piecewise')

def paged_decode_attention_op(*args, **kwargs):
    """
    Piecewise版本的paged_decode_attention
    完全透明地转发所有参数给vendor实现
    """
    return vendor_ops_registry["paged_decode_attention"](*args, **kwargs)

def fill_kv_cache_op(*args, **kwargs):
    """
    Piecewise版本的fill_kv_cache  
    完全透明地转发所有参数给vendor实现
    """
    return vendor_ops_registry["fill_kv_cache"](*args, **kwargs)

logger.info("dlinfer_piecewise ops (transparent wrapper) loaded successfully")
