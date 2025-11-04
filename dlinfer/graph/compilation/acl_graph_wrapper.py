"""
ACL Graph包装器：自动capture/replay
参考：vLLM-Ascend的ACLGraphWrapper
/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py
"""
import torch
import torch_npu
import gc
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass
from contextlib import ExitStack
from unittest.mock import patch
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.acl_graph')

@dataclass
class ACLGraphEntry:
    """ACL Graph缓存条目"""
    cache_key: tuple  # 完整的shape信息作为key
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Any = None  # 使用weak ref避免内存占用（由graph pool管理）
    input_addresses: Optional[list[int]] = None  # Debug模式用
    # Buffer管理：capture时记录的固定buffer，replay时使用
    input_buffers: Optional[list[torch.Tensor]] = None  # 固定地址的buffer
    buffer_indices: Optional[list[int]] = None  # 哪些参数需要buffer（只buffer tensor参数）
    
def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    创建tensor的weak reference（简化版）
    
    注意：这不是真正的weak ref，而是让PyTorch的graph pool管理内存
    通过detach()打断autograd图，减少引用计数
    """
    if isinstance(t, torch.Tensor):
        # detach()创建新tensor但共享storage，减少强引用
        return t.detach()
    elif isinstance(t, (list, tuple)):
        return type(t)(weak_ref_tensor(x) for x in t)
    elif isinstance(t, dict):
        return {k: weak_ref_tensor(v) for k, v in t.items()}
    else:
        return t

class AscendPiecewiseGraphWrapper(torch.nn.Module):
    """
    ACL Graph包装器
    
    功能：
    1. 首次调用某batch_size：capture ACL Graph
    2. 后续调用：replay cached graph
    3. 自动管理内存和GC
    
    注意：
    - 不负责buffer的copy，外部需保证输入tensor地址一致
    - 参考vLLM-Ascend的设计
    - 继承自 torch.nn.Module，以便可以作为 GraphModule 的子模块
    """
    
    def __init__(
        self,
        runnable: Callable,
        is_first_graph: bool = False,
        is_last_graph: bool = False,
    ):
        super().__init__()  # 调用 nn.Module 的 __init__
        
        # 简单保存 runnable，不需要复杂的序列化处理
        # 因为我们使用 __dict__ 直接赋值（参考 vLLM）
        self.runnable = runnable
        self.is_first_graph = is_first_graph
        self.is_last_graph = is_last_graph
        
        # 缓存：{batch_size: ACLGraphEntry}
        self.cache: Dict[int, ACLGraphEntry] = {}
        
        # Graph pool（全局共享）
        # torch.cuda.graph_pool_handle() 在某些版本也适用于 NPU
        try:
            self.graph_pool = torch.cuda.graph_pool_handle()
        except Exception:
            # Fallback: 不使用 graph pool
            self.graph_pool = None
        
        # Debug模式
        self.debug_mode = logger.level <= 10  # DEBUG level
        
        logger.debug(f"ACLGraphWrapper created: "
                     f"first={is_first_graph}, last={is_last_graph}")
    
    def __call__(self, *args, **kwargs):
        """
        执行：自动capture或replay
        
        首次遇到某个shape组合 → capture
        已存在 → replay
        
        注意：使用完整的input shapes作为cache key，而不仅仅是batch_size
        这是因为prefill和decode阶段的shapes不同
        """
        # 生成cache key：所有tensor参数的shape tuple
        cache_key = self._generate_cache_key(args, kwargs)
        
        if cache_key not in self.cache:
            # 首次遇到这个shape组合：capture
            logger.info(f"Capturing ACL Graph for shapes: {cache_key}")
            return self._capture(cache_key, args, kwargs)
        else:
            # 已capture：replay
            logger.debug(f"Replaying ACL Graph for shapes: {cache_key}")
            return self._replay(cache_key, args, kwargs)
    
    def _generate_cache_key(self, args, kwargs) -> tuple:
        """
        生成cache key：基于所有tensor输入的shape
        
        类似vLLM的BatchDescriptor，我们用shape tuple作为key
        """
        shapes = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shapes.append(tuple(arg.shape))
        
        # 也包含kwargs中的tensor
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                shapes.append(tuple(v.shape))
        
        return tuple(shapes)
    
    def _capture(self, cache_key: tuple, args, kwargs):
        """捕获ACL Graph"""
        entry = ACLGraphEntry(cache_key=cache_key)
        
        # Ascend NPU策略：保存所有compute图的input buffers
        # 注意：与vLLM不同，Ascend NPU的allocator可能不够"伪确定性"
        # 为了稳定性，我们保存所有的input buffers
        input_buffers = []
        buffer_indices = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                input_buffers.append(arg)
                buffer_indices.append(i)
        
        entry.input_buffers = input_buffers
        entry.buffer_indices = buffer_indices
        
        if self.is_first_graph:
            logger.info(f"First graph: saved {len(input_buffers)} input buffers")
        else:
            logger.debug(f"Compute graph: saved {len(input_buffers)} input buffers")
        
        # 记录输入地址（debug模式，用于验证allocator稳定性）
        if self.debug_mode:
            input_addresses = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_addresses.append(arg.data_ptr())
            entry.input_addresses = input_addresses
            logger.debug(f"Captured input addresses: {entry.input_addresses}")
        
        aclgraph = torch.npu.NPUGraph()
        
        with ExitStack() as stack:
            # 非第一个graph：禁用GC以加速capture
            # 参考：vLLM-Ascend的优化策略
            if not self.is_first_graph:
                stack.enter_context(patch("gc.collect", lambda: None))
                stack.enter_context(patch("torch.npu.empty_cache", lambda: None))
            
            # Capture graph
            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)
        
        entry.aclgraph = aclgraph
        entry.output = output
        self.cache[cache_key] = entry
        
        if self.is_first_graph and entry.input_buffers:
            logger.info(f"ACL Graph captured for shapes {cache_key}, "
                       f"{len(entry.input_buffers)} input buffers saved")
        else:
            logger.info(f"ACL Graph captured for shapes {cache_key} "
                       f"(relying on allocator stability)")
        
        return output
    
    def _replay(self, cache_key: tuple, args, kwargs):
        """Replay ACL Graph"""
        entry = self.cache[cache_key]
        
        # 所有compute图：copy输入到固定buffers
        if entry.input_buffers is not None and entry.buffer_indices is not None:
            for buffer_idx, arg_idx in enumerate(entry.buffer_indices):
                new_input = args[arg_idx]
                old_buffer = entry.input_buffers[buffer_idx]
                
                if new_input.shape != old_buffer.shape:
                    raise RuntimeError(
                        f"Input shape mismatch during replay! "
                        f"Expected {old_buffer.shape}, got {new_input.shape}"
                    )
                
                old_buffer.copy_(new_input)
            
            if self.debug_mode and entry.input_addresses is not None:
                current_addresses = [buf.data_ptr() for buf in entry.input_buffers]
                if current_addresses != entry.input_addresses:
                    logger.error(
                        f"Buffer addresses changed unexpectedly! "
                        f"Expected {entry.input_addresses}, got {current_addresses}"
                    )
                else:
                    logger.debug("Buffer addresses verified consistent")
        
        # Replay
        entry.aclgraph.replay()
        return entry.output
    
    
    def clear_cache(self):
        """清空缓存（测试用）"""
        self.cache.clear()
        gc.collect()
        torch.npu.empty_cache()

