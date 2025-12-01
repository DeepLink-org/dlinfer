"""
Piecewise Graph Runner
lmdeploywarmup
"""

from collections import OrderedDict
from dataclasses import dataclass
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
import dlinfer.graph
from dlinfer.graph.ascend_piecewise.piecewise_backend import (
    get_capture_batch_sizes as backend_capture_batch_sizes,
)
from dlinfer.graph.ascend_piecewise.bucket_utils import limit_capture_buckets
from dlinfer.graph.ascend_piecewise.bucket_utils import adjust_capture_batch_sizes
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from dlinfer.graph.ascend_piecewise.graph_capture_session import GraphCaptureSession
from dlinfer.graph.ascend_piecewise.utils import PiecewiseEnvConfig

from torch.profiler import record_function

logger = get_logger("dlinfer")


@dataclass
class RunnerEnvironmentConfig:
    """Centralized environment variable configuration for runner."""

    debug_capture: bool = False
    capture_sizes_override: Optional[List[int]] = None
    max_capture_graphs: Optional[int] = None
    graph_capture_sizes: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "RunnerEnvironmentConfig":
        """Create configuration from environment variables with validation."""
        debug_capture = PiecewiseEnvConfig.get_debug_capture()

        # Set logger level based on debug configuration
        if debug_capture:
            import logging
            logger.setLevel(logging.DEBUG)

        # Get capture sizes with validation
        capture_sizes_override = PiecewiseEnvConfig.get_capture_sizes()
        if capture_sizes_override is not None and not capture_sizes_override:
            logger.warning(
                "DLINFER_ASCEND_CAPTURE_SIZES provided but no valid positive integers found"
            )

        # Get max capture graphs with validation
        max_capture_graphs = PiecewiseEnvConfig.get_max_capture_graphs()

        return cls(
            debug_capture=debug_capture,
            capture_sizes_override=capture_sizes_override,
            max_capture_graphs=max_capture_graphs,
            graph_capture_sizes=PiecewiseEnvConfig.get_graph_capture_sizes(),
        )


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False



class CaptureSizeProcessor:
    """Utility class for processing capture sizes configuration."""

    @staticmethod
    def get_distributed_config() -> Optional[Any]:
        """Safely extract distributed configuration."""
        try:
            from lmdeploy.pytorch.distributed import get_dist_manager

            dist_config = get_dist_manager().current_context().dist_config
            return dist_config
        except Exception:  # pragma: no cover - dist context may not be initialized
            return None

    @staticmethod
    def process_capture_sizes(
        cache_config,
        model_config,
        env_config: RunnerEnvironmentConfig,
        default_capture_func,
    ) -> List[int]:
        """Process and validate capture sizes from multiple sources."""
        dist_config = CaptureSizeProcessor.get_distributed_config()

        # Get default capture sizes from backend
        default_capture_sizes = default_capture_func(cache_config.max_batches)

        # Use override sizes if provided, otherwise use defaults
        base_sizes = env_config.capture_sizes_override or default_capture_sizes

        # Validate override sizes against defaults
        if env_config.capture_sizes_override:
            filtered = [size for size in base_sizes if size in default_capture_sizes]
            if not filtered:
                logger.warning(
                    "DLINFER_ASCEND_CAPTURE_SIZES=%s has no overlap with default buckets %s; falling back",
                    env_config.capture_sizes_override,
                    default_capture_sizes,
                )
                base_sizes = default_capture_sizes
            else:
                base_sizes = filtered
                logger.info(
                    "Using user-specified capture buckets before adjustment: %s",
                    base_sizes,
                )

        # Adjust sizes based on configuration
        adjusted_sizes = adjust_capture_batch_sizes(
            base_sizes,
            model_config=model_config,
            cache_config=cache_config,
            dist_config=dist_config,
            logger_=logger,
        )

        return adjusted_sizes

    @staticmethod
    def apply_capture_limits(
        capture_sizes: List[int], model_config, env_config: RunnerEnvironmentConfig
    ) -> List[int]:
        """Apply limits to capture sizes if needed."""
        dist_config = CaptureSizeProcessor.get_distributed_config()

        # Only apply trimming if custom capture sizes are not provided via environment variable
        if not env_config.graph_capture_sizes:
            limited_sizes = limit_capture_buckets(
                capture_sizes,
                model_config=model_config,
                dist_config=dist_config,
                max_capture_graphs=env_config.max_capture_graphs,
            )
            if not limited_sizes:
                logger.warning(
                    "Ascend capture bucket limiter returned empty list; falling back to defaults."
                )
                limited_sizes = capture_sizes
            capture_sizes = limited_sizes

        # Ensure capture sizes are sorted ascending for minimal bucket selection
        return sorted(set(capture_sizes))


class AscendPiecewiseSingleGraphRunner:
    """GraphCaptureSession-backed runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Any,
        model_config: ModelConfig,
        device: torch.device,
    ):
        self.session = GraphCaptureSession(
            model=model,
            model_config=model_config,
            max_batches=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
        )

    @record_function("capture_cudagraph")
    def capture(self, **kwargs) -> Any:
        """Capture graph."""
        return self.session.capture(**kwargs)

    @record_function("forward_cudagraph")
    def forward(self, **kwargs) -> Any:
        """forward."""
        return self.session.forward(**kwargs)


class RunnerCache:
    """Simple helper to manage cached graph runners."""

    def __init__(self) -> None:
        self._entries: "OrderedDict[Any, Any]" = OrderedDict()

    def get_or_create(
        self, key: Any, factory: Callable[[], Any]
    ) -> Tuple[Any, bool]:
        entry = self._entries.get(key)
        if entry is None:
            runner = factory()
            self._entries[key] = runner
            return runner, True
        else:
            self._entries.move_to_end(key)
            return entry, False

    def clear(self) -> None:
        self._entries.clear()


class AscendPiecewiseGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device,
    ):
        super().__init__(model, model_config, cache_config, backend_config, device)

        # Initialize basic attributes
        self._initialize_basic_attributes()

        # Setup environment configuration
        self.env_config = RunnerEnvironmentConfig.from_environment()

        # Configure capture sizes using utility
        self._configure_capture_sizes()

    def _initialize_basic_attributes(self) -> None:
        """Initialize basic runner attributes."""
        self.max_batches = self.cache_config.max_batches
        self.max_tokens = self.cache_config.max_prefill_token_num
        self.num_blocks = self.cache_config.num_gpu_blocks
        self.enable_graph = self.check_enable_graph()
        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_cache = RunnerCache()
        self._runner_cls = AscendPiecewiseSingleGraphRunner

        # Configure global graph settings
        dlinfer.graph.config.enable_graph_mode = True
        dlinfer.graph.config.piecewise_graph_enabled = True

    def _configure_capture_sizes(self) -> None:
        """Configure capture batch sizes using utility class."""
        # Process capture sizes from multiple sources
        adjusted_sizes = CaptureSizeProcessor.process_capture_sizes(
            self.cache_config,
            self.model_config,
            self.env_config,
            backend_capture_batch_sizes,
        )

        # Apply capture limits
        self._capture_batch_sizes = CaptureSizeProcessor.apply_capture_limits(
            adjusted_sizes, self.model_config, self.env_config
        )

    
    def check_enable_graph(self) -> Callable:
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, "support_cuda_graph", _false)

    def _get_capture_tokens(self, batch_size: int) -> int:
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f"Unsupported batch_size={batch_size}"

    def get_graph_key(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[int, bool, bool]:
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        num_tokens = input_ids.numel()
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        if meta.padding_batch_size is None:
            new_num_tokens = self._get_capture_tokens(num_tokens)
        else:
            new_num_tokens = self._get_capture_tokens(meta.padding_batch_size)
        return (new_num_tokens, is_decoding, enable_microbatch)

    def __call__(self, **kwargs) -> Any:
        """Main execution entry point with graph optimization."""
        if self._should_use_eager_mode(**kwargs):
            return self._execute_eager(**kwargs)

        return self._execute_with_graph_optimization(**kwargs)

    def _should_use_eager_mode(self, **kwargs) -> bool:
        """Check if eager mode should be used instead of graph optimization."""
        return not self.enable_graph(**kwargs)

    def _execute_eager(self, **kwargs) -> Any:
        """Execute model in eager mode without graph optimization."""
        with record_function("forward_eager"):
            return self.model(**kwargs)

    def _execute_with_graph_optimization(self, **kwargs) -> Any:
        """Execute model with graph capture and reuse optimization."""
        graph_key = self.get_graph_key(**kwargs)
        runner, created = self._get_or_create_runner(graph_key)

        if created:
            return self._handle_new_runner_creation(runner, **kwargs)
        else:
            return self._handle_existing_runner_reuse(runner, **kwargs)

    def _get_or_create_runner(
        self, graph_key: Tuple[int, bool, bool]
    ) -> Tuple[Any, bool]:
        """Get existing runner from cache or create new one."""
        max_tokens, is_decoding, _ = graph_key
        max_batches = max_tokens if is_decoding else self.max_batches

        def _factory() -> Any:
            return self._runner_cls(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
            )

        return self._runner_cache.get_or_create(graph_key, _factory)

    def _handle_new_runner_creation(
        self, runner: Any, **kwargs
    ) -> Any:
        """Handle first-time graph capture for new runner."""
        from dlinfer.graph import config

        original_is_capturing = config.is_capturing
        config.is_capturing = True

        try:
            output = runner.capture(**kwargs)
            return output
        finally:
            config.is_capturing = original_is_capturing

    def _handle_existing_runner_reuse(
        self, runner: Any, **kwargs
    ) -> Any:
        """Handle graph reuse for existing runner."""
        return runner.forward(**kwargs)

    @record_function("prepare_inputs_for_generation")
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self) -> None:
        """Remove all graphs to prevent hanging on exit."""
        self._runner_cache.clear()

    def update_inputs(self, inputs) -> Any:
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            tp_size = self._get_capture_tokens(padding_batch_size)
            # Sync both tp_sizes and moe_tp_sizes so downstream MoE ops see the padded token count.
            logger.debug(
                "[AscendRunner] sync_tp_size padding_batch_size=%s tp_size=%s "
                "tp_sizes_before=%s moe_tp_sizes_before=%s",
                padding_batch_size,
                tp_size,
                dp_meta.tp_sizes,
                dp_meta.moe_tp_sizes,
            )
            dp_meta.sync_tp_size(tp_size)
            logger.debug(
                "[AscendRunner] synced tp_sizes_after=%s moe_tp_sizes_after=%s",
                dp_meta.tp_sizes,
                dp_meta.moe_tp_sizes,
            )
            if self.env_config.debug_capture:
                logger.info(
                    "[AscendRunner] synced padding_batch=%s capture_tp=%s",
                    padding_batch_size,
                    tp_size,
                )
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        return self._capture_batch_sizes
