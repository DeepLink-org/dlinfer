"""
Eager Execution Wrapper for Piecewise Mode

在 Piecewise 模式下，某些 ops（如 attention）需要执行 eager 版本。
这个 wrapper 临时关闭 graph mode，执行 eager 版本的函数。
"""
import torch
import torch.nn as nn
from typing import Any, Callable
from contextlib import contextmanager
import dlinfer.graph
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.eager_wrapper')


@contextmanager
def eager_mode():
    """
    临时关闭 graph mode 的上下文管理器
    
    用法:
        with eager_mode():
            result = some_dlinfer_op(...)  # 执行 eager 版本
    """
    original_mode = dlinfer.graph.config.enable_graph_mode
    try:
        dlinfer.graph.config.enable_graph_mode = False
        yield
    finally:
        dlinfer.graph.config.enable_graph_mode = original_mode


class EagerExecutionWrapper(nn.Module):
    """
    强制 eager 执行的 wrapper
    
    功能：
    - 将某个 op 或 module 包装，使其始终在 eager 模式下执行
    - 即使外层 enable_graph_mode=True，这个 wrapper 内部也会临时关闭
    
    使用场景：
    - Piecewise 模式中的 attention ops
    - 需要 eager 执行以便调试的 ops
    """
    
    def __init__(self, op_or_module: Callable, op_name: str = "unknown"):
        super().__init__()
        self.op_or_module = op_or_module
        self.op_name = op_name
        # logger.info(f"Created EagerExecutionWrapper for '{op_name}'")
    
    def forward(self, *args, **kwargs) -> Any:
        """
        执行时临时关闭 graph mode
        """
        with eager_mode():
            result = self.op_or_module(*args, **kwargs)
        return result
    
    def __repr__(self):
        return f"EagerExecutionWrapper({self.op_name})"


class ConditionalEagerWrapper(nn.Module):
    """
    条件性 eager 执行 wrapper
    
    根据输入特征（如 batch_size, is_decoding）决定是否执行 eager
    """
    
    def __init__(
        self, 
        op_or_module: Callable, 
        op_name: str = "unknown",
        always_eager: bool = True
    ):
        super().__init__()
        self.op_or_module = op_or_module
        self.op_name = op_name
        self.always_eager = always_eager
        # logger.debug(f"Created ConditionalEagerWrapper for '{op_name}' "
        #             f"(always_eager={always_eager})")
    
    def should_use_eager(self, *args, **kwargs) -> bool:
        """
        决定是否使用 eager 模式
        
        可以根据具体需求覆盖这个方法
        """
        if self.always_eager:
            return True
        
        # 未来可以添加更复杂的逻辑
        # 例如：根据 batch_size, is_decoding 等条件
        return False
    
    def forward(self, *args, **kwargs) -> Any:
        """执行"""
        if self.should_use_eager(*args, **kwargs):
            with eager_mode():
                result = self.op_or_module(*args, **kwargs)
        else:
            result = self.op_or_module(*args, **kwargs)
        return result
    
    def __repr__(self):
        return f"ConditionalEagerWrapper({self.op_name}, always_eager={self.always_eager})"


def wrap_for_eager_execution(runnable: Callable, name: str = "unknown") -> nn.Module:
    """
    便捷函数：将 callable 包装为强制 eager 执行的 module
    
    Args:
        runnable: 要包装的 callable（op 或 module）
        name: 名称（用于日志）
    
    Returns:
        EagerExecutionWrapper 实例
    """
    return EagerExecutionWrapper(runnable, name)

