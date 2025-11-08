"""
Eager Execution Wrapper for Piecewise Mode

In piecewise mode, certain operations (like attention) require eager execution.
This wrapper temporarily disables graph mode to execute eager versions of functions.
"""
import torch.nn as nn
from typing import Any, Callable
from contextlib import contextmanager
import dlinfer.graph

@contextmanager
def eager_mode():
    """
    Context manager to temporarily disable graph mode.

    Usage:
        with eager_mode():
            result = some_dlinfer_op(...)  # Execute eager version
    """
    original_mode = dlinfer.graph.config.enable_graph_mode
    try:
        dlinfer.graph.config.enable_graph_mode = False
        yield
    finally:
        dlinfer.graph.config.enable_graph_mode = original_mode

class EagerExecutionWrapper(nn.Module):
    """
    Wrapper to force eager execution.

    Features:
    - Wraps operations or modules to always execute in eager mode
    - Temporarily disables graph mode internally even when outer enable_graph_mode=True
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

