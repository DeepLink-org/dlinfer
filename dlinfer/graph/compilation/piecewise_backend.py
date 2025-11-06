"""
Dlinfer Piecewise Backend for torch.compile

新方案（基于 enable_graph_mode=True）：
1. 启用 enable_graph_mode，使用 torch.ops.dlinfer::xxx（已注册的 custom ops）
2. dynamo 可以追踪完整图（不会因为未注册的 ops 报错）
3. 识别 attention ops（torch.ops.dlinfer.paged_decode_attention）
4. 分割图后：
   - attention 子图：用 EagerExecutionWrapper 包装（临时关闭 enable_graph_mode）
   - compute 子图：用 ACL Graph 包装

参考：
- vLLM VllmBackend: /vllm-workspace/vllm/vllm/compilation/backends.py:401-613
- vLLM PiecewiseCompileInterpreter: /vllm-workspace/vllm/vllm/compilation/backends.py:286-379
"""
import torch
import torch.fx as fx
import functools
from typing import Callable, List
from lmdeploy.utils import get_logger
from .graph_splitter import split_graph, SplitItem
from .acl_graph_wrapper import AscendPiecewiseGraphWrapper
from .eager_wrapper import EagerExecutionWrapper

logger = get_logger('dlinfer.backend')

# 全局 graph_pool 实例，确保多个实例共用一个 graph_pool
_global_graph_pool = None

def get_graph_pool():
    """获取全局唯一的 graph_pool 实例"""
    global _global_graph_pool
    if _global_graph_pool is None:
        _global_graph_pool = torch.cuda.graph_pool_handle()
        logger.info("Created global graph pool for shared use across instances")
    return _global_graph_pool

# Batch Size管理策略（从ascend_cudagraph.py移植）
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n

def get_ascend_compatible_size(n: int):
    """Get ascend compatible size. 参考ascend_cudagraph.py实现"""
    if n <= 16:
        n = next_power_of_2(n)
    elif n <= 256:
        n = (n + 15) & ~0xF
    else:
        n = (((n - 1) >> 8) + 1) << 8
    return n

@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size. 从ascend_cudagraph.py移植"""
    ret = []
    batch_size = 1
    batch_step_1, batch_step_2 = 16, 256
    # power of 2
    while batch_size <= min(batch_step_1, max_batches):
        ret.append(batch_size)
        batch_size *= 2

    # step 1
    ret += list(range(batch_size, min(max_batches, batch_step_2) + 1, batch_step_1))

    # step 2
    ret += list(range(ret[-1] + batch_step_2, max_batches + 1, batch_step_2))

    # ensure max_batches in ret
    if max_batches != ret[-1]:
        ret.append(max_batches)

    return ret


def get_capture_batch_sizes(max_batches: int) -> List[int]:
    """获取Ascend兼容的捕获batch size列表"""
    return _get_capture_batch_size_impl(max_batches)

class DlinferPiecewiseBackend:
    """
    Dlinfer自定义torch.compile backend
    
    参考：vLLM的VllmBackend
    /vllm-workspace/vllm/vllm/compilation/backends.py:401
    
    功能：
    1. 接收dynamo trace的FX graph
    2. 按splitting_ops分割图
    3. 为非attention部分添加ACL Graph wrapper
    4. 返回可执行的split_gm
    """
    
    # Splitting ops列表
    # 注意：这些是 torch.ops.dlinfer::xxx 注册的 ops（通过 register_custom_op）
    # 在 FX graph 中，str(node.target) 的格式是 'dlinfer.xxx'
    # 包括 attention 相关的所有 ops：
    # - dlinfer.prefill_attention (prefill 阶段)
    # - dlinfer.paged_decode_attention (decode 阶段)
    # - dlinfer.fill_kv_cache (KV cache 更新)
    SPLITTING_OPS = [
        # prefill_attention (prefill 阶段的 attention)
        "dlinfer.prefill_attention",
        # paged_decode_attention (decode 阶段的 attention)
        "dlinfer.paged_decode_attention",
        # incre_flash_attention（也是 decode 阶段的 attention，旧版本可能用这个）
        "dlinfer.incre_flash_attention",
        # fill_kv_cache（KV cache 更新，属于 attention 流程）
        "dlinfer.fill_kv_cache",
        # paged_prefill_attention (paged prefill)
        "dlinfer.paged_prefill_attention",
    ]
    
    def __init__(self):
        # 每次调用 backend 都会重新 split/ wrap
        pass

    def __call__(self, gm: fx.GraphModule, example_inputs) -> Callable:
        """
        Backend入口
        
        Args:
            gm: Dynamo trace的FX graph
            example_inputs: 示例输入（fake tensors）
        
        Returns:
            split_gm: 可执行的分割后的GraphModule
        
        注意：此方法可能被 PyTorch 多次调用（不同输入shapes）
        但对于我们的场景，只有第一次调用会进行分割和包装
        """
        try:
            # # Step 0: 打印完整的 FX graph（调试用，只在第一次打印）
            # if self._compilation_count == 1:
            #     logger.info("=" * 60)
            #     logger.info("Original FX Graph:")
            #     logger.info("=" * 60)
            #     logger.info(gm.graph)
            #     logger.info("=" * 60)
            #     logger.info("Detailed node information:")
            #     for node in gm.graph.nodes:
            #         if node.op == 'call_function':
            #             logger.info(f"  Node: {node.name}, op: {node.op}, target: {node.target}, target_str: '{str(node.target)}'")
            #     logger.info("=" * 60)
            
            # Step 1: 分割图（每次都重新split）
            # 注意：每个batch size的FX graph可能不同（硬编码shape不同）
            # 所以必须每次都重新split，不能复用！
            logger.info("Step 1: Splitting graph...")
            split_gm, split_items = split_graph(gm, self.SPLITTING_OPS)
            
            logger.info(f"Graph split into {len(split_items)} submodules:")
            # for i, item in enumerate(split_items):
            #     graph_type = "ATTENTION" if item.is_splitting_graph else "COMPUTE"
            #     logger.info(f"  [{i}] {item.submod_name}: {graph_type}")
            
            # Step 2: 包装子图（每次都重新wrap以获得独立的cache）
            # 参考 vLLM 的实现（backends.py:366-375）
            # 使用 __dict__ 直接赋值，避免 PyTorch Module 的序列化机制
            logger.info("Step 2: Wrapping submodules...")
            for item in split_items:
                submod_name = item.submod_name
                original_submod = getattr(split_gm, submod_name)

                if item.is_splitting_graph:
                    wrapped = EagerExecutionWrapper(
                        op_or_module=original_submod,
                        op_name=f"attention_{submod_name}"
                    )
                else:
                    is_first = item.graph_id == 0
                    is_last = item.graph_id == len(split_items) - 1

                    wrapped = AscendPiecewiseGraphWrapper(
                        runnable=original_submod,
                        is_first_graph=is_first,
                        is_last_graph=is_last,
                        graph_pool=get_graph_pool(),
                    )

                split_gm.__dict__[submod_name] = wrapped
            
            logger.info("Step 3: Graph preparation complete")
            logger.info("=" * 60)
            
            # Step 3: 返回split_gm
            return split_gm
        
        except Exception as e:
            logger.error(f"Error in DlinferPiecewiseBackend: {e}", exc_info=True)
            raise
    
    def reset(self):
        """重置状态（测试用）"""
        pass

# 全局backend实例
# 注意：每次compile都应该创建新实例
def create_backend():
    """创建backend实例"""
    return DlinferPiecewiseBackend()

