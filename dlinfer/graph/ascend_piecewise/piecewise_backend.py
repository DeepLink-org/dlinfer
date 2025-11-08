"""
Dlinfer Piecewise Backend for torch.compile.

Strategy: Separate attention operations (executed eagerly) from compute operations 
(optimized with ACL Graph) using enable_graph_mode=True.
"""
import torch
import torch.fx as fx
import functools
from typing import Callable, List, Any, Dict, Tuple, Union
from lmdeploy.utils import get_logger
from dlinfer.graph.ascend_piecewise.graph_splitter import split_graph, SplitItem
from dlinfer.graph.ascend_piecewise.acl_graph_wrapper import AscendPiecewiseGraphWrapper
from dlinfer.graph.ascend_piecewise.eager_wrapper import EagerExecutionWrapper
from dlinfer.graph.ascend_piecewise.utils import (
    debug_fx_graph_structure,
    debug_fx_graph_nodes,
    debug_graph_splitting,
    debug_submodule_wrapping,
    debug_compilation_summary,
    is_fx_graph_debug_enabled,
)

logger = get_logger('dlinfer.backend')

def is_debug_enabled() -> bool:
    """Check if FX graph debugging is enabled via environment variable."""
    import os
    return os.environ.get('DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG', '0') == '1'

_global_graph_pool = None

def get_graph_pool() -> Any:
    """Get the global unique graph_pool instance."""
    global _global_graph_pool
    if _global_graph_pool is None:
        _global_graph_pool = torch.cuda.graph_pool_handle()
        if is_debug_enabled():
            logger.info("Created global graph pool for shared use across instances")
    return _global_graph_pool

def next_power_of_2(n: int) -> int:
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

def get_ascend_compatible_size(n: int) -> int:
    return next_power_of_2(n)

@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int) -> List[int]:
    """Generate capture batch size sequence."""
    ret = []
    batch_size = 1
    while batch_size < max_batches:
        ret.append(batch_size)
        batch_size *= 2
    ret.append(max_batches)
    return ret

def get_capture_batch_sizes(max_batches: int) -> List[int]:
    """Get Ascend-compatible capture batch size list."""
    return _get_capture_batch_size_impl(max_batches)

class DlinferPiecewiseBackend:
    """
    Custom torch.compile backend for Dlinfer with piecewise graph optimization.

    Reference: vLLM VllmBackend (vllm/vllm/compilation/backends.py)

    Features:
    1. Receives dynamo-traced FX graphs
    2. Splits graphs by splitting operations
    3. Wraps non-attention parts with ACL Graph
    4. Returns executable split GraphModule
    """
    
    SPLITTING_OPS = [
        "dlinfer.prefill_attention",
        "dlinfer.paged_decode_attention",
        "dlinfer.incre_flash_attention",
        "dlinfer.fill_kv_cache",
        "dlinfer.paged_prefill_attention",
    ]

    def __init__(self):
        self._compilation_count = 0
    
    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Callable[..., Any]:
        """
        Backend entry point.

        Args:
            gm: Dynamo-traced FX graph
            example_inputs: Example inputs (fake tensors)

        Returns:
            split_gm: Executable split GraphModule

        """
        if not isinstance(gm, fx.GraphModule):
            raise TypeError("gm must be a torch.fx.GraphModule instance")
        if not example_inputs:
            raise ValueError("example_inputs cannot be empty")

        # Increment compilation count for debugging
        self._compilation_count += 1

        try:
            # Debug Step 0: Print original FX graph structure (only on first compilation)
            if self._compilation_count == 1 and is_fx_graph_debug_enabled():
                debug_fx_graph_structure(gm, "Original FX Graph (First Compilation)")
                debug_fx_graph_nodes(gm, filter_ops=['call_function'], title="Original FX Graph - Call Functions")

            if is_debug_enabled():
                logger.info("Step 1: Splitting graph...")
            split_gm, split_items = split_graph(gm, self.SPLITTING_OPS)

            if not split_items:
                raise RuntimeError("Graph splitting produced no submodules")

            if is_debug_enabled():
                logger.info("Graph split into %d submodules", len(split_items))

            # Debug Step 1: Analyze graph splitting results
            if is_fx_graph_debug_enabled():
                debug_graph_splitting(split_items, split_gm)

            if is_debug_enabled():
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

            # Debug Step 2: Analyze submodule wrapping
            if is_fx_graph_debug_enabled():
                debug_submodule_wrapping(split_items, split_gm)

            if is_debug_enabled():
                logger.info("Step 3: Graph preparation complete")
                logger.info("=" * 60)

            # Debug Step 3: Final compilation summary
            if is_fx_graph_debug_enabled():
                debug_compilation_summary(gm, split_gm, split_items, self._compilation_count)

            return split_gm
        
        except Exception as e:
            logger.error(f"Error in DlinferPiecewiseBackend: {e}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Reset state (for testing)."""
        pass

def create_backend():
    return DlinferPiecewiseBackend()

