"""
Piecewise专用算子 - 简化版
直接包装 vendor ops，不需要复杂的 torch.library 注册
"""
import torch
from dlinfer.vendor.ascend.torch_npu_ops import vendor_ops_registry
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.piecewise')

# 直接定义函数，让 torch.compile 无法内联它们

def paged_decode_attention_op(*args, **kwargs):
    """Piecewise版本的paged_decode_attention - 直接调用vendor实现"""
    return vendor_ops_registry["paged_decode_attention"](*args, **kwargs)

def fill_kv_cache_op(*args, **kwargs):
    """Piecewise版本的fill_kv_cache - 直接调用vendor实现"""
    return vendor_ops_registry["fill_kv_cache"](*args, **kwargs)

logger.info("dlinfer_piecewise ops (simple wrapper) loaded successfully")

