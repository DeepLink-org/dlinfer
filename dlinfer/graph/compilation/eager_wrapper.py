"""
Eager Execution Wrapper for Piecewise Mode

在 Piecewise 模式下，某些 ops（如 attention）需要执行 eager 版本。
这个 wrapper 临时关闭 graph mode，执行 eager 版本的函数。
"""
import torch.nn as nn
from typing import Any, Callable
from contextlib import contextmanager
import dlinfer.graph


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
    """
    
    def __init__(self, op_or_module: Callable, op_name: str = "unknown"):
        super().__init__()
        self.op_or_module = op_or_module
        self.op_name = op_name
    
    def forward(self, *args, **kwargs) -> Any:
        with eager_mode():
            return self.op_or_module(*args, **kwargs)
    
    def __repr__(self):
        return f"EagerExecutionWrapper({self.op_name})"

