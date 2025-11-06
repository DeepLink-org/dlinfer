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

# 导入batch size管理函数
try:
    from .piecewise_backend import get_ascend_compatible_size
except ImportError:
    # Fallback实现
    def get_ascend_compatible_size(n: int):
        """Fallback implementation of ascend compatible size"""
        if n <= 16:
            # power of 2
            power = 1
            while power < n:
                power *= 2
            return power
        else:
            # round up to multiple of 16
            return ((n + 15) // 16) * 16

logger = get_logger('dlinfer.acl_graph')

@dataclass
class ACLGraphEntry:
    """ACL Graph缓存条目"""
    cache_key: tuple  # 完整的shape信息作为key
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Any = None  # 使用weak ref避免内存占用（由graph pool管理）
    input_addresses: Optional[list[int]] = None  # Debug模式用
    arg_buffers: Optional[list[torch.Tensor]] = None
    arg_indices: Optional[list[int]] = None
    arg_shapes: Optional[list[torch.Size]] = None
    arg_views: Optional[list[torch.Tensor]] = None


def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    创建tensor的weak reference（简化版）
    
    注意：这不是真正的weak ref，而是让PyTorch的graph pool管理内存
    通过detach()打断autograd图，减少引用计数
    """
    if isinstance(t, torch.Tensor):
        # 关键修复：移除contiguous()，保持原始内存布局避免精度问题
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
    - 缓存逻辑移至AscendPiecewiseGraphRunner层
    """
    
    def __init__(
        self,
        runnable: Callable,
        is_first_graph: bool = False,
        is_last_graph: bool = False,
        graph_pool = None,
        is_decoding: bool = None,  # 添加阶段参数，None表示自动检测
    ):
        super().__init__()  # 调用 nn.Module 的 __init__

        # 简单保存 runnable，不需要复杂的序列化处理
        # 因为我们使用 __dict__ 直接赋值（参考 vLLM）
        self.runnable = runnable
        self.is_first_graph = is_first_graph
        self.is_last_graph = is_last_graph

        # 关键添加：阶段检测逻辑
        self.is_decoding = is_decoding
        self.auto_detect_stage = is_decoding is None
        self._use_graph = True if self.auto_detect_stage else bool(self.is_decoding)
        
        # Graph pool（全局共享）
        # 使用全局 graph_pool 实例，确保多个实例共用一个 graph_pool
        self.graph_pool = graph_pool
        assert self.graph_pool is not None
        
        # Debug模式
        self.debug_mode = logger.level <= 10  # DEBUG level
        
        # 缓存：{shape_key: ACLGraphEntry}
        self.cache: Dict[tuple, ACLGraphEntry] = {}

        # Debug模式（修复重复设置）
        self.debug_mode = logger.level <= 10  # DEBUG level

        # 记录decode阶段的基准shape，用于强制单形状graph
        self._canonical_signature: Optional[tuple] = None

        logger.debug(f"ACLGraphWrapper created: "
                     f"first={is_first_graph}, last={is_last_graph}, "
                     f"is_decoding={is_decoding}, auto_detect={self.auto_detect_stage}")

    def _build_signature(self, args, kwargs) -> tuple:
        """根据当前输入构建 shape 签名（仅在 decode 阶段记录一次）。"""
        signature = []
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                signature.append((f"arg{idx}", tuple(arg.shape)))

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                signature.append((f"kw:{key}", tuple(value.shape)))

        return tuple(signature)

    def _ensure_signature(self, args, kwargs):
        """确保 decode 阶段的 shape 与首次 capture 完全一致。"""
        signature = self._build_signature(args, kwargs)
        stored = self._canonical_signature

        if stored is None:
            self._canonical_signature = signature
            if self.debug_mode:
                logger.debug("Recorded canonical decode signature: %s", signature)
        elif signature != stored:
            raise RuntimeError(
                "Input shapes changed between captures; expected %s, got %s. "
                "AscendPiecewiseGraphWrapper only supports a single shape per stage." % (stored, signature)
            )

        return signature
    
    def __call__(self, *args, **kwargs):
        """
        执行：自动capture或replay
        当前实现仅支持decode阶段进入图。
        """
        if not self._use_graph:
            if self.debug_mode:
                logger.debug("Non-decode stage: using eager execution")
            return self.runnable(*args, **kwargs)

        # Decode阶段：使用graph模式
        cache_key = self._generate_cache_key(args, kwargs)
        stage_signature = self._canonical_signature

        entry = self.cache.get(cache_key)
        if entry is None:
            logger.info(
                "Decode stage: capturing ACL Graph (signature=%s)",
                stage_signature,
            )
            return self._capture(cache_key, args, kwargs)

        logger.debug(
            "Decode stage: replaying ACL Graph (signature=%s)",
            stage_signature,
        )
        return self._replay(cache_key, args, kwargs)
    
    def _generate_cache_key(self, args, kwargs) -> tuple:
        """生成用于缓存的简单 key，并强制复用首个 capture 的形状。"""
        # 仅按阶段区分：decode阶段固定 key
        return (True,)
    
    def _capture(self, cache_key: tuple, args, kwargs):
        """捕获ACL Graph"""
        self._ensure_signature(args, kwargs)
        entry = ACLGraphEntry(cache_key=cache_key)
        entry.arg_buffers = []
        entry.arg_indices = []
        entry.arg_shapes = []
        entry.arg_views = []

        new_args = list(args)

        for idx, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue

            shape = arg.shape
            entry.arg_indices.append(idx)
            entry.arg_shapes.append(shape)

            if arg.dim() == 0:
                buffer = torch.empty_like(arg, device=arg.device)
                buffer.copy_(arg)
                view = buffer
            else:
                max_batch = get_ascend_compatible_size(shape[0])
                buffer_shape = (max_batch,) + tuple(shape[1:])
                buffer = torch.empty(buffer_shape, device=arg.device, dtype=arg.dtype)
                view = buffer[: shape[0]]
                view.copy_(arg)

            new_args[idx] = view
            entry.arg_buffers.append(buffer)
            entry.arg_views.append(view)

        if self.debug_mode and entry.arg_views:
            entry.input_addresses = [view.data_ptr() for view in entry.arg_views]
            logger.debug(
                "Captured arg buffer addresses for %s: %s",
                cache_key,
                entry.input_addresses,
            )

        aclgraph = torch.npu.NPUGraph()
        with ExitStack() as stack:
            # 非第一个graph：禁用GC以加速capture
            # 参考：vLLM-Ascend的优化策略
            if not self.is_first_graph:
                stack.enter_context(patch("gc.collect", lambda: None))
                stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

            # Capture graph：positional tensor已经映射到持久化缓冲区
            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                output = self.runnable(*tuple(new_args), **kwargs)

        entry.aclgraph = aclgraph
        entry.output = weak_ref_tensor(output)
        self.cache[cache_key] = entry

        logger.info(f"ACL Graph captured for shapes {cache_key}")

        return output
    
    def _replay(self, cache_key: tuple, args, kwargs):
        """Replay ACL Graph"""
        entry = self.cache[cache_key]

        if self.debug_mode and entry.input_addresses:
            new_addresses = [view.data_ptr() for view in entry.arg_views]
            if new_addresses != entry.input_addresses:
                logger.error(
                    "Arg buffer addresses changed for cache_key=%s\nExpected: %s\nGot: %s",
                    cache_key,
                    entry.input_addresses,
                    new_addresses,
                )
                raise RuntimeError("Input buffer addresses changed between capture and replay")

        # Replay 之前，把新的输入数据拷贝到缓冲区
        for arg_idx, buffer, target_view, expected_shape in zip(
            entry.arg_indices or [],
            entry.arg_buffers or [],
            entry.arg_views or [],
            entry.arg_shapes or [],
        ):
            new_tensor = args[arg_idx]

            if new_tensor.shape != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch for positional arg{arg_idx}: expected {expected_shape}, got {new_tensor.shape}"
                )
            if new_tensor.device != buffer.device:
                raise RuntimeError(
                    f"Device mismatch for positional arg{arg_idx}: expected {buffer.device}, got {new_tensor.device}"
                )

            if new_tensor.data_ptr() != target_view.data_ptr():
                target_view.copy_(new_tensor)

        # Replay
        entry.aclgraph.replay()
        return entry.output
    
    def reset(self):
        """重置graph（测试用）"""
        self.cache.clear()
        logger.debug("ACL Graph cache reset")
    
    def clear_cache(self):
        """清空缓存（测试用）"""
        # 优化的缓存清理：先清理graph资源
        for _, entry in self.cache.items():
            if entry.aclgraph is not None:
                del entry.aclgraph
            entry.output = None
            entry.arg_buffers = None
            entry.arg_views = None

        self.cache.clear()
        gc.collect()
        torch.npu.empty_cache()

        logger.info("ACL Graph cache cleared with optimized resource cleanup")
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'cache') and self.cache:
                for entry in self.cache.values():
                    if entry.aclgraph is not None:
                        del entry.aclgraph
                self.cache.clear()
        except Exception:
            pass

