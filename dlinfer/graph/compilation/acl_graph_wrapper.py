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
        
        # Graph pool（全局共享）
        # 使用全局 graph_pool 实例，确保多个实例共用一个 graph_pool
        self.graph_pool = graph_pool
        if self.graph_pool is None:
            from .piecewise_backend import get_graph_pool
            self.graph_pool = get_graph_pool()
            logger.debug("Using global graph pool")
        else:
            logger.debug("Using provided graph pool")
        
        # Debug模式
        self.debug_mode = logger.level <= 10  # DEBUG level
        
        # 缓存：{shape_key: ACLGraphEntry}
        self.cache: Dict[tuple, ACLGraphEntry] = {}

        # Debug模式（修复重复设置）
        self.debug_mode = logger.level <= 10  # DEBUG level

        # 记录不同阶段（prefill/decode）的基准shape，用于强制单形状graph
        self._canonical_signatures: Dict[bool, tuple] = {}

        logger.debug(f"ACLGraphWrapper created: "
                     f"first={is_first_graph}, last={is_last_graph}, "
                     f"is_decoding={is_decoding}, auto_detect={self.auto_detect_stage}")

    def _detect_stage(self, args, kwargs) -> bool:
        """
        检测当前执行阶段（prefill vs decode）
        参考ascend_cudagraph.py的get_graph_key()逻辑
        """
        if not self.auto_detect_stage:
            return self.is_decoding

        # 1. 检查 attn_metadata.is_decoding（最可靠）
        attn_metadata = kwargs.get('attn_metadata', None)
        if attn_metadata is not None and hasattr(attn_metadata, 'is_decoding'):
            is_decoding = getattr(attn_metadata, 'is_decoding', False)
            logger.debug(f"Stage detected via attn_metadata.is_decoding: {is_decoding}")
            return is_decoding

        # 2. 检查输入tensor的形状特征（fallback）
        # Prefill阶段：较长的input_ids，decode阶段：单个token
        input_ids = kwargs.get('input_ids', None)
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                batch_size, seq_len = input_ids.shape
                is_decoding = seq_len == 1  # decode阶段通常seq_len=1
                logger.debug(f"Stage detected via input_ids shape: {input_ids.shape} -> decode={is_decoding}")
                return is_decoding

        # 3. 检查args中的tensor
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
                batch_size, seq_len = arg.shape[:2]
                if seq_len == 1:
                    logger.debug(f"Stage detected via tensor shape: {arg.shape} -> decode=True")
                    return True
                else:
                    logger.debug(f"Stage detected via tensor shape: {arg.shape} -> decode=False")
                    return False

        # 默认假设为prefill阶段（更安全）
        logger.debug("Stage detection fallback: assuming prefill (decode=False)")
        return False

    def _should_use_graph_mode(self, args, kwargs) -> bool:
        """
        判断是否应该使用graph模式
        根据删除的ascend_piecewise_runner.py逻辑：
        - Prefill阶段：使用eager模式（shape变化大）
        - Decode阶段：使用graph模式（shape固定）
        """
        is_decoding = self._detect_stage(args, kwargs)

        should_use_graph = is_decoding  # 只有decode阶段使用graph

        if self.debug_mode:
            logger.debug(f"Graph mode decision: is_decoding={is_decoding}, "
                        f"should_use_graph={should_use_graph}")

        return should_use_graph

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

    def _ensure_signature(self, is_decoding: bool, args, kwargs):
        """确保 decode 阶段的 shape 与首次 capture 完全一致。"""
        signature = self._build_signature(args, kwargs)
        stored = self._canonical_signatures.get(is_decoding)

        if stored is None:
            self._canonical_signatures[is_decoding] = signature
            if self.debug_mode:
                logger.debug("Recorded canonical signature for stage %s: %s", is_decoding, signature)
        elif signature != stored:
            raise RuntimeError(
                "Input shapes changed between captures; expected %s, got %s. "
                "AscendPiecewiseGraphWrapper only supports a single shape per stage." % (stored, signature)
            )

        return signature
    
    def __call__(self, *args, **kwargs):
        """
        执行：自动capture或replay，支持prefill/decode阶段切换

        关键修复：根据删除的ascend_piecewise_runner.py逻辑
        - Prefill阶段：使用eager模式，不走graph
        - Decode阶段：使用graph模式，自动capture/replay
        """
        # 首先检测执行阶段
        should_use_graph = self._should_use_graph_mode(args, kwargs)

        if not should_use_graph:
            # Prefill阶段：使用eager模式，避免graph capture
            logger.debug("Prefill stage: using eager execution (skipping graph)")
            return self.runnable(*args, **kwargs)

        # Decode阶段：使用graph模式
        # 生成cache key：包含shape和阶段信息
        cache_key = self._generate_cache_key(args, kwargs)
        stage_signature = self._canonical_signatures.get(cache_key[0])

        if cache_key not in self.cache:
            # 首次遇到这个shape组合：capture
            logger.info(
                "Decode stage: capturing ACL Graph (signature=%s)",
                stage_signature,
            )
            return self._capture(cache_key, args, kwargs)
        else:
            # 已capture：replay
            logger.debug(
                "Decode stage: replaying ACL Graph (signature=%s)",
                stage_signature,
            )
            return self._replay(cache_key, args, kwargs)
    
    def _generate_cache_key(self, args, kwargs) -> tuple:
        """生成用于缓存的简单 key，并强制复用首个 capture 的形状。"""
        is_decoding = self._detect_stage(args, kwargs)
        self._ensure_signature(is_decoding, args, kwargs)
        # 仅按阶段区分：decode 阶段固定为 True
        return (is_decoding,)
    
    def _capture(self, cache_key: tuple, args, kwargs):
        """捕获ACL Graph"""
        entry = ACLGraphEntry(cache_key=cache_key)

        new_args = list(args)
        arg_buffers: list[torch.Tensor] = []
        arg_indices: list[int] = []
        arg_shapes: list[torch.Size] = []
        arg_views: list[torch.Tensor] = []

        for idx, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue

            if arg.dim() == 0:
                buffer = torch.empty_like(arg, device=arg.device)
                buffer.copy_(arg)
                new_args[idx] = buffer
                arg_buffers.append(buffer)
                arg_indices.append(idx)
                arg_shapes.append(arg.shape)
                arg_views.append(buffer)
                continue

            batch_size = arg.shape[0]
            max_batch = get_ascend_compatible_size(batch_size)
            tail_shape = tuple(arg.shape[1:])
            buffer_shape = (max_batch,) + tail_shape
            full_buffer = torch.empty(buffer_shape, device=arg.device, dtype=arg.dtype)
            view = full_buffer[:batch_size]
            view.copy_(arg)
            new_args[idx] = view
            arg_buffers.append(full_buffer)
            arg_indices.append(idx)
            arg_shapes.append(arg.shape)
            arg_views.append(view)

        entry.arg_buffers = arg_buffers if arg_buffers else None
        entry.arg_indices = arg_indices if arg_indices else None
        entry.arg_shapes = arg_shapes if arg_shapes else None
        entry.arg_views = arg_views if arg_views else None

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

        if entry.arg_buffers and entry.arg_indices:
            for buffer_idx, arg_idx in enumerate(entry.arg_indices):
                buffer = entry.arg_buffers[buffer_idx]
                expected_shape = (
                    entry.arg_shapes[buffer_idx]
                    if entry.arg_shapes and buffer_idx < len(entry.arg_shapes)
                    else buffer.shape
                )

                new_tensor = args[arg_idx]

                if new_tensor.shape != expected_shape:
                    raise RuntimeError(
                        f"Shape mismatch for positional arg{arg_idx}: expected {expected_shape}, got {new_tensor.shape}"
                    )

                if new_tensor.device != buffer.device:
                    raise RuntimeError(
                        f"Device mismatch for positional arg{arg_idx}: expected {buffer.device}, got {new_tensor.device}"
                    )

                if len(expected_shape) == 0:
                    target_view = (
                        entry.arg_views[buffer_idx]
                        if entry.arg_views and buffer_idx < len(entry.arg_views)
                        else buffer
                    )
                    if new_tensor.data_ptr() != target_view.data_ptr():
                        target_view.copy_(new_tensor)
                    continue

                batch_size = expected_shape[0]
                target_view = (
                    entry.arg_views[buffer_idx]
                    if entry.arg_views and buffer_idx < len(entry.arg_views)
                    else buffer
                )
                if target_view.shape[0] != batch_size:
                    target_view = buffer[(slice(0, batch_size),) + tuple(slice(None) for _ in expected_shape[1:])]

                if new_tensor.data_ptr() != target_view.data_ptr():
                    target_view.copy_(new_tensor)

        if self.debug_mode and entry.input_addresses is not None and entry.arg_views:
            new_addresses = [view.data_ptr() for view in entry.arg_views]
            if new_addresses != entry.input_addresses:
                logger.error(
                    "Arg buffer addresses changed for cache_key=%s\nExpected: %s\nGot: %s",
                    cache_key,
                    entry.input_addresses,
                    new_addresses,
                )
                raise RuntimeError("Input buffer addresses changed between capture and replay")

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

