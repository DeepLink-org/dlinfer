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
from dlinfer.graph.ascend_piecewise.utils import (
    is_acl_graph_debug_enabled,
    is_debug_enabled,
)

logger = get_logger("dlinfer.acl_graph")


@dataclass
class ACLGraphStats:
    """Simple counter for active ACL graphs."""

    active_graphs: int = 0

    def increment(self) -> int:
        """Increment active graph count and return new total."""
        self.active_graphs += 1

        if is_acl_graph_debug_enabled():
            logger.info("[ACLGraphCapture] Active ACL graphs: %s", self.active_graphs)

        return self.active_graphs


# Global statistics instance
_global_stats = ACLGraphStats()


def increment_active_graphs() -> int:
    """Increment global active ACL graph count."""
    return _global_stats.increment()


def clear_active_graphs() -> None:
    """Clear global active ACL graph count."""
    _global_stats.active_graphs = 0


def get_active_graphs_count() -> int:
    """Get current active ACL graph count."""
    return _global_stats.active_graphs


class InputValidator:
    """Unified input validation utilities for ACL Graph operations."""

    @staticmethod
    def build_signature(
        args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """Build shape signature from inputs for consistency validation."""
        signature = []

        # Process positional arguments
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                signature.append((f"arg{idx}", tuple(arg.shape)))

        # Process keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                signature.append((f"kw:{key}", tuple(value.shape)))

        return tuple(signature)

    @staticmethod
    def validate_tensor_consistency(
        arg_indices: List[int], arg_shapes: List[torch.Size], args: Tuple[Any, ...]
    ) -> None:
        """Validate that input tensors match expected shapes."""
        for arg_idx, expected_shape in zip(arg_indices, arg_shapes):
            new_tensor = args[arg_idx]
            if new_tensor.shape != expected_shape:
                ExceptionHandler.handle_shape_mismatch(
                    arg_idx, expected_shape, new_tensor.shape
                )

    @staticmethod
    def validate_device_consistency(
        arg_indices: List[int], arg_views: List[torch.Tensor], args: Tuple[Any, ...]
    ) -> None:
        """Validate that input tensors are on expected devices."""
        for arg_idx, target_view in zip(arg_indices, arg_views):
            new_tensor = args[arg_idx]
            if new_tensor.device != target_view.device:
                ExceptionHandler.handle_device_mismatch(
                    arg_idx, target_view.device, new_tensor.device
                )

    @staticmethod
    def find_buffers_to_copy(
        arg_indices: List[int], arg_views: List[torch.Tensor], args: Tuple[Any, ...]
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        """Find tensors that need data copying due to address changes."""
        buffers_to_copy = []

        for arg_idx, target_view in zip(arg_indices, arg_views):
            new_tensor = args[arg_idx]
            if new_tensor.data_ptr() != target_view.data_ptr():
                buffers_to_copy.append((arg_idx, target_view, new_tensor))

        return buffers_to_copy

    @staticmethod
    def copy_tensor_data(
        buffers_to_copy: List[Tuple[int, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Copy tensor data for all buffers that need updating."""
        for arg_idx, target_view, new_tensor in buffers_to_copy:
            target_view.copy_(new_tensor)


class MemoryManager:
    """Unified memory management for ACL Graph operations."""

    @staticmethod
    def cleanup_cache_entry(entry, cache_key: Tuple[Any, ...] = None) -> None:
        """Safely clean up a single cache entry."""
        try:
            # Clean up ACL Graph
            if entry.acl_graph is not None:
                del entry.acl_graph
                entry.acl_graph = None
        except Exception as e:
            key_info = f" for cache {cache_key}" if cache_key else ""
            logger.warning(f"Failed to delete ACL graph{key_info}: {e}")

        # Clean up references
        entry.output = None
        entry.arg_buffers = None
        entry.arg_views = None

    @staticmethod
    def cleanup_all_caches(cache_dict) -> None:
        """Clean up all cache entries and clear the cache."""
        for cache_key, entry in cache_dict.items():
            MemoryManager.cleanup_cache_entry(entry, cache_key)

        cache_dict.clear()

    @staticmethod
    def force_garbage_collection() -> None:
        """Force garbage collection and NPU cache cleanup."""
        try:
            gc.collect()
            torch.npu.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to cleanup memory during garbage collection: {e}")


class ExceptionHandler:
    """Unified exception handling utilities for ACL Graph operations."""

    @staticmethod
    def handle_shape_mismatch(
        arg_idx: int, expected_shape: torch.Size, actual_shape: torch.Size
    ) -> None:
        """Handle tensor shape mismatch with standardized error."""
        raise RuntimeError(
            f"Shape mismatch for arg{arg_idx}: expected {expected_shape}, got {actual_shape}"
        )

    @staticmethod
    def handle_device_mismatch(
        arg_idx: int, expected_device: torch.device, actual_device: torch.device
    ) -> None:
        """Handle tensor device mismatch with standardized error."""
        raise RuntimeError(
            f"Device mismatch for arg{arg_idx}: expected {expected_device}, got {actual_device}"
        )

    @staticmethod
    def handle_signature_change(
        expected_signature: Tuple, actual_signature: Tuple
    ) -> None:
        """Handle input signature change with standardized error."""
        raise RuntimeError(
            f"Input shapes changed between captures; expected {expected_signature}, got {actual_signature}. "
            "AscendPiecewiseGraphWrapper only supports a single shape per stage."
        )

    @staticmethod
    def handle_address_change() -> None:
        """Handle buffer address change with standardized error."""
        raise RuntimeError("Input buffer addresses changed between capture and replay")

    @staticmethod
    def handle_execution_error(original_error: Exception) -> None:
        """Handle ACL graph execution errors with standardized error."""
        error_msg = f"Error in ACL graph execution: {str(original_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from original_error

    @staticmethod
    def handle_capture_error(
        original_error: Exception,
        cache_key: Tuple[Any, ...],
        acl_graph: Optional[torch.npu.NPUGraph] = None,
    ) -> None:
        """Handle ACL graph capture errors with cleanup."""
        if acl_graph is not None:
            try:
                del acl_graph
            except Exception:
                pass  # Ignore cleanup errors

        logger.error(
            f"Failed to capture ACL graph for shapes {cache_key}: {original_error}"
        )
        raise


class DebugLogger:
    """Unified debug logging utility for ACL Graph operations."""

    def __init__(self, logger_instance: Any):
        self.logger = logger_instance
        self.debug_mode = False
        self.acl_debug = False
        self.env_debug = False
        self.env_acl_debug = False

    def configure(self, debug_mode: bool = False, acl_debug: bool = False):
        """Configure debug modes."""
        self.debug_mode = debug_mode
        self.acl_debug = acl_debug
        self.env_debug = is_debug_enabled()
        self.env_acl_debug = is_acl_graph_debug_enabled()

    def debug(self, message: str, *args) -> None:
        """Log debug message if any debug mode is enabled."""
        if self.debug_mode or self.env_debug:
            self.logger.debug(message, *args)

    def acl_debug(self, message: str, *args) -> None:
        """Log ACL debug message if ACL debug is enabled."""
        if self.acl_debug or self.env_acl_debug:
            self.logger.debug(message, *args)

    def acl_info(self, message: str, *args) -> None:
        """Log ACL info message if ACL debug is enabled."""
        if self.acl_debug or self.env_acl_debug:
            self.logger.info(message, *args)

    def env_info(self, message: str, *args) -> None:
        """Log info message if environment debug is enabled."""
        if self.env_debug:
            self.logger.info(message, *args)

    def conditional_debug(self, condition: bool, message: str, *args) -> None:
        """Log debug message only if condition and debug mode are both True."""
        if condition and (self.debug_mode or self.env_debug):
            self.logger.debug(message, *args)


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


def weak_ref_tensor(t) -> Any:
    """
    Create a detached tensor reference for efficient graph memory management.

    This is not a true weak reference but allows PyTorch's graph pool
    to manage memory more efficiently by detaching from autograd.
    """
    # Fast path: handle the most common case first
    if isinstance(t, torch.Tensor):
        return t.detach()

    # Handle container types efficiently
    if isinstance(t, (list, tuple)):
        return type(t)(weak_ref_tensor(x) for x in t)

    if isinstance(t, dict):
        return {k: weak_ref_tensor(v) for k, v in t.items()}

    # Return as-is for non-tensor types
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

    # Single graph cache key - each wrapper instance only handles one graph mode
    _SINGLE_GRAPH_CACHE_KEY = (True,)

    def __init__(
        self,
        runnable: Callable[..., Any],
        is_first_graph: bool = False,
        is_last_graph: bool = False,
        graph_pool: Optional[Any] = None,
        is_decoding: Optional[bool] = None,
    ):
        if not callable(runnable):
            raise ValueError("runnable must be callable")
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

        # Initialize unified debug logger
        self.debug_logger = DebugLogger(logger)
        debug_mode = is_debug_enabled()
        acl_debug = is_acl_graph_debug_enabled()
        self.debug_logger.configure(debug_mode=debug_mode, acl_debug=acl_debug)

        # Keep backward compatibility
        self.debug_mode = debug_mode
        self.acl_debug = acl_debug

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
        return InputValidator.build_signature(args, kwargs)

    def _ensure_signature(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """Ensure decode stage shapes match first capture exactly."""
        signature = self._build_signature(args, kwargs)
        stored = self._canonical_signature

        if stored is None:
            self._canonical_signature = signature
            self.debug_logger.debug(
                "Recorded canonical decode signature: %s", signature
            )
        elif signature != stored:
            ExceptionHandler.handle_signature_change(stored, signature)

        return signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute: automatic capture or replay.
        Current implementation only supports decode stage graph execution.
        """
        try:
            if not self._use_graph:
                self.debug_logger.debug("Non-decode stage: using eager execution")
                return self.runnable(*args, **kwargs)

            if not self.cache:
                self.debug_logger.env_info(
                    "Decode stage: capturing ACL Graph (signature=%s)",
                    self._canonical_signature,
                )
                return self._capture(self._SINGLE_GRAPH_CACHE_KEY, args, kwargs)

            self.debug_logger.debug(
                "Decode stage: replaying ACL Graph (signature=%s)",
                self._canonical_signature,
            )
            return self._replay(self._SINGLE_GRAPH_CACHE_KEY, args, kwargs)

        except Exception as e:
            ExceptionHandler.handle_execution_error(e)

    def _capture(
        self, cache_key: Tuple[Any, ...], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Capture ACL Graph for given inputs."""
        self._ensure_signature(args, kwargs)
        entry = ACLGraphEntry(cache_key=cache_key)

        # Prepare entry with tensor metadata
        self._prepare_capture_entry(entry, args, cache_key)

        # Execute graph capture with error handling
        try:
            ref_output = self._execute_graph_capture(entry, args, kwargs)
            # Complete capture
            self._complete_capture(entry, cache_key, ref_output)
            return ref_output
        except Exception as e:
            # Get the ACL graph for cleanup if it exists
            acl_graph = getattr(entry, "acl_graph", None)
            ExceptionHandler.handle_capture_error(e, cache_key, acl_graph)

    def _prepare_capture_entry(
        self, entry: ACLGraphEntry, args: Tuple[Any, ...], cache_key: Tuple[Any, ...]
    ) -> None:
        """Prepare ACLGraphEntry with tensor metadata for capture."""
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

            if self.acl_debug:
                entry.input_addresses = [view.data_ptr() for view in views]
                self.debug_logger.acl_debug(
                    "Captured arg buffer addresses for %s: %s",
                    cache_key,
                    entry.input_addresses,
                )
        else:
            entry.arg_indices = []
            entry.arg_shapes = []
            entry.arg_views = []
            entry.arg_buffers = []
            if self.acl_debug:
                entry.input_addresses = []

    def _execute_graph_capture(
        self, entry: ACLGraphEntry, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Execute the actual graph capture process."""
        from dlinfer.graph import config

        original_is_capturing = config.is_capturing
        config.is_capturing = True

        acl_graph = torch.npu.NPUGraph()

        try:
            with ExitStack() as stack:
                if not self.is_first_graph:
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                with torch.npu.graph(
                    acl_graph,
                    pool=self.graph_pool,
                ):
                    output = self.runnable(*args, **kwargs)

            entry.acl_graph = acl_graph
            ref_output = weak_ref_tensor(output)
            entry.output = ref_output

            return ref_output

        finally:
            config.is_capturing = original_is_capturing

    def _complete_capture(
        self, entry: ACLGraphEntry, cache_key: Tuple[Any, ...], ref_output: Any
    ) -> None:
        """Complete the capture process with caching and logging."""
        self._add_to_cache(cache_key, entry)
        increment_active_graphs()

        if self.acl_debug:
            active_count = get_active_graphs_count()
            self.debug_logger.acl_info("ACL Graph captured for shapes %s", cache_key)
            self.debug_logger.acl_info("Total ACL Graph captures: %s", active_count)

    def _replay(
        self, cache_key: Tuple[Any, ...], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Replay captured ACL Graph."""
        entry = self.cache.pop(cache_key)
        self.cache[cache_key] = entry

        if self.acl_debug:
            self._validate_input_buffers(entry, cache_key)
            self._copy_input_data(entry, args, cache_key)

        entry.acl_graph.replay()
        return entry.output

    def _add_to_cache(self, cache_key: Tuple[Any, ...], entry: ACLGraphEntry) -> None:
        if cache_key not in self.cache:
            self.cache[cache_key] = entry
        self.debug_logger.debug(
            "Added cache entry %s (cache size: %d)", cache_key, len(self.cache)
        )

    def _validate_input_buffers(
        self, entry: ACLGraphEntry, cache_key: Tuple[Any, ...]
    ) -> None:
        """Validate input buffer addresses in debug mode"""
        self.debug_logger.conditional_debug(
            self.debug_mode and entry.input_addresses and entry.arg_views,
            "Validating input buffer addresses for cache_key=%s",
            cache_key,
        )

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
            ExceptionHandler.handle_address_change()

    def _copy_input_data(
        self, entry: ACLGraphEntry, args: Tuple[Any, ...], cache_key: Tuple[Any, ...]
    ) -> None:
        if not entry.arg_indices:
            return

        # Validate tensor shapes and device consistency using unified validator
        InputValidator.validate_tensor_consistency(
            entry.arg_indices, entry.arg_shapes, args
        )
        InputValidator.validate_device_consistency(
            entry.arg_indices, entry.arg_views, args
        )

        # Find buffers that need copying
        buffers_to_copy = InputValidator.find_buffers_to_copy(
            entry.arg_indices, entry.arg_views, args
        )

        if buffers_to_copy:
            self.debug_logger.env_info(
                "Copying %d changed buffer addresses for cache_key=%s",
                len(buffers_to_copy),
                cache_key,
            )
            for arg_idx, target_view, new_tensor in buffers_to_copy:
                self.debug_logger.debug(
                    "Copying arg%d: expected 0x%x, got 0x%x",
                    arg_idx,
                    target_view.data_ptr(),
                    new_tensor.data_ptr(),
                )

            # Copy all tensor data in batch
            InputValidator.copy_tensor_data(buffers_to_copy)

    def reset(self) -> None:
        """Reset graph cache (for testing)."""
        self.cache.clear()
        self.debug_logger.debug("ACL Graph cache reset")

    def clear_cache(self) -> None:
        # Clean up all cache entries
        MemoryManager.cleanup_all_caches(self.cache)

        # Force garbage collection
        MemoryManager.force_garbage_collection()

        self.debug_logger.env_info("ACL Graph cache cleared")

    def __del__(self) -> None:
        try:
            if hasattr(self, "cache") and self.cache:
                MemoryManager.cleanup_all_caches(self.cache)
        except Exception:
            # Silently ignore errors in destructor
            pass