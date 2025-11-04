"""
Piecewise 专用包装函数

策略：使用 @torch._dynamo.disable 阻止 torch.compile 追踪进这些函数
这样可以：
1. 避免 torch.compile 参数匹配错误
2. 让这些函数作为 graph break point（虽然不是理想的 splitting point，但可以工作）
3. Attention 部分会在 eager 模式下执行，符合我们的目标
"""
import torch
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.piecewise')

# 延迟导入，避免循环依赖
_llm_ops = None

def _get_llm_ops():
    """延迟导入 dlinfer.ops.llm，避免循环依赖"""
    global _llm_ops
    if _llm_ops is None:
        from dlinfer.ops import llm as _llm_ops_module
        _llm_ops = _llm_ops_module
    return _llm_ops

@torch._dynamo.disable
def paged_decode_attention_wrapper(*args, **kwargs):
    """
    转发到原始的 dlinfer.ops.llm.paged_decode_attention
    
    使用 @torch._dynamo.disable 阻止 torch.compile 追踪进这个函数。
    这会导致 graph break，但可以确保：
    1. 避免参数匹配错误
    2. Attention 在 eager 模式下执行
    3. 其他部分仍然可以被编译优化（虽然可能不是完整的 ACL Graph）
    """
    llm_ops = _get_llm_ops()
    return llm_ops.paged_decode_attention(*args, **kwargs)

@torch._dynamo.disable
def fill_kv_cache_wrapper(*args, **kwargs):
    """
    转发到原始的 dlinfer.ops.llm.fill_kv_cache
    
    使用 @torch._dynamo.disable 阻止 torch.compile 追踪。
    """
    llm_ops = _get_llm_ops()
    return llm_ops.fill_kv_cache(*args, **kwargs)

# 使用不同的导出名避免与 torch.ops 冲突
paged_decode_attention_op = paged_decode_attention_wrapper
fill_kv_cache_op = fill_kv_cache_wrapper

logger.info("Piecewise wrappers loaded (plain Python functions)")

