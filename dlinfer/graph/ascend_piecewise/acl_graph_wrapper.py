"""
Ascend Piecewise Graph Wrapper with automatic capture/replay functionality.

Reference implementation: vLLM-Ascend ACLGraphWrapper
vllm-ascend/vllm_ascend/compilation/acl_graph.py
"""

import torch
import torch_npu
import gc
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from contextlib import ExitStack
from unittest.mock import patch
from collections import OrderedDict
from lmdeploy.utils import get_logger

logger = get_logger("dlinfer.acl_graph")


def is_debug_enabled() -> bool:
    """Check if ACL graph debugging is enabled via environment variable."""
    import os

    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


# Global counter for ACL graph capture statistics
acl_graph_capture_count = 0

# Default maximum number of cached graphs to prevent memory bloat
DEFAULT_MAX_CACHE_SIZE = 8


@dataclass
class ACLGraphEntry:
    """ACL Graph cache entry storing captured graph metadata and buffers."""

    cache_key: Tuple[Any, ...]
    acl_graph: Optional[torch.npu.NPUGraph] = None
    output: Any = None
    input_addresses: Optional[List[int]] = None
    arg_buffers: Optional[List[torch.Tensor]] = None
    arg_indices: Optional[List[int]] = None
    arg_shapes: Optional[List[torch.Size]] = None
    arg_views: Optional[List[torch.Tensor]] = None


def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Create a detached tensor reference for efficient graph memory management.

    This is not a true weak reference but allows PyTorch's graph pool
    to manage memory more efficiently by detaching from autograd.
    """
    if isinstance(t, torch.Tensor):
        return t.detach()
    elif isinstance(t, (list, tuple)):
        return type(t)(weak_ref_tensor(x) for x in t)
    elif isinstance(t, dict):
        return {k: weak_ref_tensor(v) for k, v in t.items()}
    else:
        return t


class AscendPiecewiseGraphWrapper(torch.nn.Module):
    """
    Ascend ACL Graph wrapper with automatic capture/replay functionality.

    Features:
    1. First call with specific batch_size: captures ACL Graph
    2. Subsequent calls: replays cached graph for performance
    3. Automatic memory management and garbage collection

    Notes:
    - External code must ensure input tensor addresses remain consistent
    - Based on vLLM-Ascend design principles
    - Inherits from torch.nn.Module for GraphModule compatibility
    - Caching logic handled at AscendPiecewiseGraphRunner level
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        is_first_graph: bool = False,
        is_last_graph: bool = False,
        graph_pool: Optional[Any] = None,
        is_decoding: Optional[bool] = None,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        if not callable(runnable):
            raise ValueError("runnable must be callable")
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        if graph_pool is None:
            raise ValueError("graph_pool cannot be None")

        super().__init__()

        self.runnable = runnable
        self.is_first_graph = is_first_graph
        self.is_last_graph = is_last_graph

        self.is_decoding = is_decoding
        self.auto_detect_stage = is_decoding is None
        self._use_graph = self.auto_detect_stage or bool(is_decoding)

        self.graph_pool = graph_pool
        self.debug_mode = logger.level <= 10

        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[Tuple[Any, ...], ACLGraphEntry] = OrderedDict()

        self._canonical_signature: Optional[Tuple[Tuple[str, Tuple[int, ...]], ...]] = (
            None
        )

        logger.debug(
            "ACLGraphWrapper created: first=%s, last=%s, is_decoding=%s, auto_detect=%s",
            is_first_graph,
            is_last_graph,
            is_decoding,
            self.auto_detect_stage,
        )

    def _build_signature(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """Build shape signature from current inputs (recorded once in decode stage)."""
        signature = []
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                signature.append((f"arg{idx}", tuple(arg.shape)))

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                signature.append((f"kw:{key}", tuple(value.shape)))

        return tuple(signature)

    def _ensure_signature(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """Ensure decode stage shapes match first capture exactly."""
        signature = self._build_signature(args, kwargs)
        stored = self._canonical_signature

        if stored is None:
            self._canonical_signature = signature
            if self.debug_mode:
                logger.debug("Recorded canonical decode signature: %s", signature)
        elif signature != stored:
            raise RuntimeError(
                "Input shapes changed between captures; expected %s, got %s. "
                "AscendPiecewiseGraphWrapper only supports a single shape per stage."
                % (stored, signature)
            )

        return signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute: automatic capture or replay.
        Current implementation only supports decode stage graph execution.
        """
        try:
            if not self._use_graph:
                self._debug_log("Non-decode stage: using eager execution")
                return self.runnable(*args, **kwargs)

            cache_key = self._generate_cache_key(args, kwargs)

            if cache_key not in self.cache:
                if is_debug_enabled():
                    logger.info(
                        "Decode stage: capturing ACL Graph (signature=%s)",
                        self._canonical_signature,
                    )
                return self._capture(cache_key, args, kwargs)

            self._debug_log(
                "Decode stage: replaying ACL Graph (signature=%s)",
                self._canonical_signature,
            )
            return self._replay(cache_key, args, kwargs)

        except Exception as e:
            logger.error("Error in ACL graph execution: %s", str(e))
            raise RuntimeError(f"Error in ACL graph execution: {str(e)}") from e

    def _debug_log(self, message: str, *args: Any) -> None:
        """Helper method for debug logging to avoid conditional checks"""
        if self.debug_mode:
            logger.debug(message, *args)

    def _generate_cache_key(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[bool, ...]:
        return (True,)

    def _capture(
        self, cache_key: Tuple[Any, ...], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Capture ACL Graph for given inputs."""
        global acl_graph_capture_count
        self._ensure_signature(args, kwargs)
        entry = ACLGraphEntry(cache_key=cache_key)

        tensor_args = [
            (idx, arg) for idx, arg in enumerate(args) if isinstance(arg, torch.Tensor)
        ]

        if tensor_args:
            indices, shapes, views, buffers = zip(
                *[(idx, arg.shape, arg, arg) for idx, arg in tensor_args]
            )

            entry.arg_indices = list(indices)
            entry.arg_shapes = list(shapes)
            entry.arg_views = list(views)
            entry.arg_buffers = list(buffers)

            if self.debug_mode:
                entry.input_addresses = [view.data_ptr() for view in views]
                logger.debug(
                    "Captured arg buffer addresses for %s: %s",
                    cache_key,
                    entry.input_addresses,
                )
        else:
            entry.arg_indices = []
            entry.arg_shapes = []
            entry.arg_views = []
            entry.arg_buffers = []

        from dlinfer.graph import config

        original_is_capturing = config.is_capturing
        config.is_capturing = True

        # current_stream = torch.cuda.current_stream()
        acl_graph = torch.npu.NPUGraph()

        try:
            with ExitStack() as stack:
                if not self.is_first_graph:
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                with torch.npu.graph(
                    acl_graph,
                    pool=self.graph_pool,
                    # stream=current_stream,
                    # auto_dispatch_capture=True,
                ):
                    output = self.runnable(*args, **kwargs)

            entry.acl_graph = acl_graph
            # entry.output = weak_ref_tensor(output)
            ref_output = weak_ref_tensor(output)
            entry.output = ref_output

            self._add_to_cache(cache_key, entry)
            acl_graph_capture_count += 1

            if is_debug_enabled():
                logger.info(f"########## ACL Graph captured for shapes {cache_key}")
                logger.info(f"########## Total ACL Graph captures: {acl_graph_capture_count}")
            print(f"########## Total ACL Graph captures: {acl_graph_capture_count}")

            # return output
            return ref_output

        except Exception as e:
            if acl_graph is not None:
                del acl_graph
            logger.error("Failed to capture ACL graph for shapes %s: %s", cache_key, e)
            raise
        finally:
            config.is_capturing = original_is_capturing

    def _replay(
        self, cache_key: Tuple[Any, ...], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Replay captured ACL Graph."""
        entry = self.cache.pop(cache_key)
        self.cache[cache_key] = entry

        self._validate_input_buffers(entry, cache_key)
        self._copy_input_data(entry, args, cache_key)

        entry.acl_graph.replay()
        return entry.output

    def _add_to_cache(self, cache_key: Tuple[Any, ...], entry: ACLGraphEntry) -> None:
        if len(self.cache) >= self.max_cache_size and cache_key not in self.cache:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            if is_debug_enabled():
                logger.debug("Evicted cache entry %s due to size limit", oldest_key)

            if oldest_entry.acl_graph is not None:
                del oldest_entry.acl_graph
            oldest_entry.output = None
            oldest_entry.arg_buffers = None
            oldest_entry.arg_views = None

        self.cache[cache_key] = entry
        if is_debug_enabled():
            logger.debug(
                "Added cache entry %s (cache size: %d)", cache_key, len(self.cache)
            )

    def _validate_input_buffers(
        self, entry: ACLGraphEntry, cache_key: Tuple[Any, ...]
    ) -> None:
        """Validate input buffer addresses in debug mode"""
        if not (self.debug_mode and entry.input_addresses and entry.arg_views):
            return

        new_addresses = [view.data_ptr() for view in entry.arg_views]
        if new_addresses != entry.input_addresses:
            logger.error(
                "Arg buffer addresses changed for cache_key=%s\nExpected: %s\nGot: %s",
                cache_key,
                entry.input_addresses,
                new_addresses,
            )
            raise RuntimeError(
                "Input buffer addresses changed between capture and replay"
            )

    def _copy_input_data(
        self, entry: ACLGraphEntry, args: Tuple[Any, ...], cache_key: Tuple[Any, ...]
    ) -> None:
        if not entry.arg_indices:
            return

        for arg_idx, expected_shape in zip(entry.arg_indices, entry.arg_shapes):
            new_tensor = args[arg_idx]
            if new_tensor.shape != expected_shape:
                raise RuntimeError(
                    "Shape mismatch for arg%d: expected %s, got %s"
                    % (arg_idx, expected_shape, new_tensor.shape)
                )

        buffers_to_copy = []
        for arg_idx, target_view in zip(entry.arg_indices, entry.arg_views):
            new_tensor = args[arg_idx]

            if new_tensor.device != target_view.device:
                raise RuntimeError(
                    "Device mismatch for arg%d: expected %s, got %s"
                    % (arg_idx, target_view.device, new_tensor.device)
                )

            if new_tensor.data_ptr() != target_view.data_ptr():
                buffers_to_copy.append((arg_idx, target_view, new_tensor))

        if buffers_to_copy:
            if is_debug_enabled():
                logger.info(
                    "Copying %d changed buffer addresses for cache_key=%s",
                    len(buffers_to_copy),
                    cache_key,
                )
            for arg_idx, target_view, new_tensor in buffers_to_copy:
                if is_debug_enabled():
                    logger.debug(
                        "Copying arg%d: expected 0x%x, got 0x%x",
                        arg_idx,
                        target_view.data_ptr(),
                        new_tensor.data_ptr(),
                    )
                target_view.copy_(new_tensor)

    def reset(self) -> None:
        """Reset graph cache (for testing)."""
        self.cache.clear()
        if is_debug_enabled():
            logger.debug("ACL Graph cache reset")

    def clear_cache(self) -> None:
        for cache_key, entry in self.cache.items():
            try:
                if entry.acl_graph is not None:
                    del entry.acl_graph
            except Exception as e:
                logger.warning(f"Failed to delete ACL graph for cache {cache_key}: {e}")

            entry.output = None
            entry.arg_buffers = None
            entry.arg_views = None

        self.cache.clear()

        try:
            gc.collect()
            torch.npu.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to cleanup memory during clear_cache: {e}")

        if is_debug_enabled():
            logger.info("ACL Graph cache cleared")

    def __del__(self) -> None:
        try:
            if hasattr(self, "cache") and self.cache:
                for entry in self.cache.values():
                    if entry.acl_graph is not None:
                        del entry.acl_graph
                self.cache.clear()
        except Exception:
            pass
