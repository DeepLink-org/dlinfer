"""Graph capture session for Ascend piecewise graph mode."""

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from .utils import is_debug_enabled
from .piecewise_backend import create_backend, get_ascend_compatible_size

logger = get_logger("dlinfer.ascend.capture")
if os.environ.get("DLINFER_ASCEND_DEBUG_CAPTURE", "0") == "1":
    logger.setLevel(logging.INFO)

BuffType = Dict[str, Tensor]


@dataclass
class AscendPiecewiseGraphMeta(CudaGraphMeta):
    """Metadata for piecewise graph optimization."""

    max_batchs: int
    max_tokens: int
    num_blocks: int
    is_decoding: int
    device: torch.device
    head_dim: int = 0
    num_attention_heads: int = 0
    dtype: torch.dtype = torch.float16


class AscendPiecewiseAttentionBuffer:
    class_attention_output: Tensor = None

    @classmethod
    def get_attention_output(
        cls, batch_size: int, num_attention_heads, head_dim, dtype, device
    ) -> Tensor:
        if cls.class_attention_output is None:
            from lmdeploy.pytorch.distributed import get_tp_world_rank

            tp, tp_rank = get_tp_world_rank("attn")
            if is_debug_enabled():
                logger.info("get_attention_output: tp=%s, tp_rank=%s", tp, tp_rank)
            cls.class_attention_output = torch.empty(
                batch_size,
                num_attention_heads // tp,
                head_dim,
                dtype=dtype,
                device=device,
            )
            return cls.class_attention_output
        return cls.class_attention_output[:batch_size]


def _as_tensor_on_device(value: Any, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Convert value to tensor on the desired device/dtype without reallocating."""
    if torch.is_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(device=device, dtype=dtype)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def make_buffers_cudagraph(graph_meta: CudaGraphMeta, *args, **kwargs) -> BuffType:
    max_batches = graph_meta.max_batchs
    max_tokens = graph_meta.max_tokens
    num_blocks = graph_meta.num_blocks
    num_attention_heads = graph_meta.num_attention_heads
    head_dim = graph_meta.head_dim
    dtype = graph_meta.dtype
    device = graph_meta.device
    input_buffers: BuffType = dict()
    input_buffers["input_ids"] = torch.empty(
        1, max_tokens, dtype=torch.int32, device=device
    )

    input_buffers["position_ids"] = torch.empty(
        (1, max_tokens), dtype=torch.int32, device=device
    )

    input_buffers["block_offsets"] = torch.zeros(
        (max_batches, num_blocks), dtype=torch.int32, device=device
    )

    input_buffers["q_seqlens"] = torch.zeros(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["kv_seqlens"] = torch.zeros(
        max_batches,
        dtype=torch.int32,
    )

    input_buffers["fill_seqlens"] = torch.zeros(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["q_start_loc"] = torch.zeros(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches), dtype=torch.int64, device=device
    )
    output_buffers: BuffType = dict()
    output_buffers["attention_output"] = (
        AscendPiecewiseAttentionBuffer.get_attention_output(
            max_batches, num_attention_heads, head_dim, dtype, device
        )
    )
    return input_buffers, output_buffers


def _root_key_from_path(path: str) -> str:
    root = path.split(".", 1)[0]
    root = root.split("[", 1)[0]
    return root


def _ensure_tensor_view(
    graph_meta: CudaGraphMeta,
    name: str,
    tensor: torch.Tensor,
    target_first_dim: Optional[int] = None,
) -> torch.Tensor:
    total_shape = list(tensor.shape)
    if tensor.dim() > 0:
        desired = target_first_dim if target_first_dim is not None else total_shape[0]
        desired = max(desired, total_shape[0])
        total_shape[0] = desired

    target_shape = tuple(total_shape)
    input_buffers = graph_meta.input_buffers
    buffer = input_buffers.get(name)

    if (
        buffer is None
        or buffer.shape != target_shape
        or buffer.dtype != tensor.dtype
        or buffer.device != tensor.device
    ):
        buffer = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
        input_buffers[name] = buffer

    if tensor.dim() == 0:
        buffer.copy_(tensor)
        return buffer

    active_dim0 = tensor.shape[0]
    slices_active = [slice(0, active_dim0)] + [slice(None)] * (tensor.dim() - 1)
    buffer.zero_()
    buffer[tuple(slices_active)].copy_(tensor)

    first_dim = target_shape[0]
    view_slices = [slice(0, first_dim)] + [slice(0, s) for s in tensor.shape[1:]]
    view = buffer[tuple(view_slices)]

    return view


def _materialize_argument(
    graph_meta: CudaGraphMeta,
    key_path: str,
    value: Any,
    pad_dim: Optional[int],
) -> Any:
    root_key = _root_key_from_path(key_path)
    if root_key in {"past_key_values", "attn_metadata"}:
        return value

    if torch.is_tensor(value):
        return _ensure_tensor_view(
            graph_meta,
            f"kw:{key_path}",
            value,
            target_first_dim=pad_dim if value.dim() > 0 else None,
        )

    if isinstance(value, Mapping):
        updated = {}
        changed = False
        for sub_key, sub_val in value.items():
            new_val = _materialize_argument(
                graph_meta,
                f"{key_path}.{sub_key}",
                sub_val,
                pad_dim,
            )
            updated[sub_key] = new_val
            changed = changed or new_val is not sub_val
        return updated if changed else value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        new_items = []
        changed = False
        for idx, item in enumerate(value):
            new_item = _materialize_argument(
                graph_meta,
                f"{key_path}[{idx}]",
                item,
                pad_dim,
            )
            new_items.append(new_item)
            changed = changed or new_item is not item
        if not changed:
            return value
        return type(value)(new_items)

    return value


def fill_buffers_cudagraph(
    graph_meta: CudaGraphMeta,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: List,
    attn_metadata: Any,
    inputs_embeds: Tensor,
    **kwargs,
) -> Dict[str, Tensor]:
    block_offsets: Tensor = attn_metadata.block_offsets
    kv_seqlens: List = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers
    output_buffers: BuffType = graph_meta.output_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    if graph_meta.is_decoding:
        padded_batch_size = max(graph_meta.max_batchs, batch_size)
    else:
        padded_batch_size = get_ascend_compatible_size(batch_size)

    input_ids_buf = input_buffers["input_ids"]
    input_ids_buf[:, :num_tokens].copy_(input_ids)
    if num_tokens < padded_batch_size:
        input_ids_buf[:, num_tokens:padded_batch_size].zero_()

    position_ids_buf = input_buffers["position_ids"]
    position_ids_buf[:, :num_tokens].copy_(position_ids)
    if num_tokens < padded_batch_size:
        position_ids_buf[:, num_tokens:padded_batch_size].zero_()

    block_offsets_buf = input_buffers["block_offsets"]
    block_offsets_buf[:batch_size, :num_blocks].copy_(block_offsets)
    if batch_size < padded_batch_size:
        block_offsets_buf[batch_size:padded_batch_size, :num_blocks].zero_()

    kv_seqlens_buffer = input_buffers["kv_seqlens"]
    kv_seqlens_tensor = _as_tensor_on_device(
        kv_seqlens,
        dtype=kv_seqlens_buffer.dtype,
        device=kv_seqlens_buffer.device,
    )
    kv_seqlens_buffer[:batch_size].copy_(kv_seqlens_tensor)
    if batch_size < padded_batch_size:
        kv_seqlens_buffer[batch_size:padded_batch_size].zero_()

    kv_start_buffer = input_buffers["kv_start_indices"]
    kv_start_indices_tensor = _as_tensor_on_device(
        kv_start_indices,
        dtype=kv_start_buffer.dtype,
        device=kv_start_buffer.device,
    )
    kv_start_buffer[:batch_size].copy_(kv_start_indices_tensor)
    if batch_size < padded_batch_size:
        kv_start_buffer[batch_size:padded_batch_size].zero_()

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        inputs_embed_buf = input_buffers["inputs_embeds"]
        inputs_embed_buf[:, :num_tokens].copy_(inputs_embeds)
        if num_tokens < padded_batch_size:
            inputs_embed_buf[:, num_tokens:padded_batch_size].zero_()

    attn_metadata.block_offsets = input_buffers["block_offsets"][:padded_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:padded_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][
        :padded_batch_size
    ]
    attn_metadata.attn_output_buffer = output_buffers["attention_output"][
        :padded_batch_size
    ]

    q_seqlens_tensor = getattr(attn_metadata, "q_seqlens", None)
    if q_seqlens_tensor is not None:
        attn_metadata.q_seqlens = _ensure_tensor_view(
            graph_meta,
            "q_seqlens",
            q_seqlens_tensor,
            target_first_dim=padded_batch_size,
        )

    q_start_loc_tensor = getattr(attn_metadata, "q_start_loc", None)
    if q_start_loc_tensor is not None:
        pad_dim = (
            padded_batch_size
            if q_start_loc_tensor.dim() == 0
            else max(padded_batch_size + 1, q_start_loc_tensor.shape[0])
        )
        attn_metadata.q_start_loc = _ensure_tensor_view(
            graph_meta,
            "q_start_loc",
            q_start_loc_tensor,
            target_first_dim=pad_dim,
        )

    fill_seqlens_tensor = getattr(attn_metadata, "fill_seqlens", None)
    if fill_seqlens_tensor is not None:
        attn_metadata.fill_seqlens = _ensure_tensor_view(
            graph_meta,
            "fill_seqlens",
            fill_seqlens_tensor,
            target_first_dim=padded_batch_size,
        )

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"][:, :padded_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :padded_batch_size]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][
            :, :padded_batch_size
        ]

    handled_keys = {
        "input_ids",
        "position_ids",
        "inputs_embeds",
        "attn_metadata",
        "past_key_values",
    }

    extra_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in handled_keys:
            continue
        materialized = _materialize_argument(
            graph_meta,
            key,
            value,
            pad_dim=padded_batch_size,
        )
        extra_kwargs[key] = materialized

    new_inputs.update(extra_kwargs)

    return new_inputs


def update_context_cudagraph(graph_meta, context):
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]


class SessionProfiler:
    """Utility that tracks capture/replay timing and optional logging."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("DLINFER_ASCEND_PROFILE", "0") == "1"
        interval_env = os.environ.get("DLINFER_ASCEND_PROFILE_INTERVAL")
        try:
            self.interval = max(1, int(interval_env)) if interval_env else 100
        except ValueError:
            logger.warning(
                "Invalid DLINFER_ASCEND_PROFILE_INTERVAL=%s, using default 100",
                interval_env,
            )
            self.interval = 100

        self.capture_count = 0
        self.replay_count = 0
        self.capture_time_total = 0.0
        self.replay_time_total = 0.0
        self.stage_totals: Dict[str, float] = defaultdict(float)

    def stage_timer(self, name: str):
        """Context manager that accumulates timing for an individual stage."""

        class _StageCtx:
            def __enter__(ctx_inner):
                ctx_inner._start = time.perf_counter()

            def __exit__(ctx_inner, exc_type, exc, tb):
                elapsed = time.perf_counter() - ctx_inner._start
                self.stage_totals[name] += elapsed

        return _StageCtx()

    def record_capture(self, elapsed: float, padded_batch: int) -> None:
        """Record capture timing and optionally log."""
        self.capture_count += 1
        self.capture_time_total += elapsed
        if self.enabled and self.capture_count % self.interval == 0:
            logger.info(
                "[CaptureProfile] count=%s padded_batch=%s total=%.3fms fill_ms=%.3f model_ms=%.3f",
                self.capture_count,
                padded_batch,
                elapsed * 1000,
                self._avg_stage("capture.fill_buffers", self.capture_count),
                self._avg_stage("capture.compiled_model", self.capture_count),
            )

    def record_replay(self, elapsed: float, padded_batch: int) -> None:
        """Record replay timing and optionally log."""
        self.replay_count += 1
        self.replay_time_total += elapsed
        if self.enabled and self.replay_count % self.interval == 0:
            logger.info(
                "[ForwardProfile] reuse=%s padded_batch=%s total=%.3fms fill_ms=%.3f model_ms=%.3f",
                self.replay_count,
                padded_batch,
                elapsed * 1000,
                self._avg_stage("forward.fill_buffers", self.replay_count),
                self._avg_stage("forward.compiled_model", self.replay_count),
            )

    def _avg_stage(self, name: str, denom: int) -> float:
        if denom == 0:
            return 0.0
        total = self.stage_totals.get(name, 0.0)
        return (total * 1000) / denom


class GraphCaptureSession:
    """Encapsulates capture/replay buffers and compiled model."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        *,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.ctx_mgr = model.ctx_mgr
        self._compile_miss_count = 0
        self._last_padded_batch: Optional[int] = None
        self._profiler = SessionProfiler()

        self.meta = AscendPiecewiseGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            head_dim=self.model_config.head_dim,
            num_attention_heads=self.model_config.num_attention_heads,
            dtype=self.model_config.dtype,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
        )

        self.device = device
        self._compiled_model = None
        self._backend = None

    @record_function("ascend_piecewise_capture")
    def capture(self, **kwargs):
        """Compile and capture graph for the first time."""
        if is_debug_enabled():
            logger.info("Capturing graph with meta: %s", self.meta)

        t_start = time.perf_counter()
        num_tokens = kwargs["input_ids"].size(-1)
        self._ensure_backend()
        self.meta.input_buffers, self.meta.output_buffers = make_buffers_cudagraph(
            self.meta, **kwargs
        )

        with self._profiler.stage_timer("capture.fill_buffers"):
            padded_kwargs = fill_buffers_cudagraph(self.meta, **kwargs)
            if os.environ.get("DLINFER_ASCEND_DEBUG_CAPTURE", "0") == "1":
                input_ids = padded_kwargs.get("input_ids")
                attn_metadata = padded_kwargs.get("attn_metadata")
                meta = getattr(attn_metadata, "kv_seqlens", None)
                logger.info(
                    "[CaptureDebug] padded input_ids=%s kv_seqlens=%s q_seqlens=%s",
                    tuple(input_ids.shape) if input_ids is not None else None,
                    tuple(meta.shape) if meta is not None else None,
                    getattr(attn_metadata, "q_seqlens", None),
                )
            context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        self._last_padded_batch = context.block_offsets.shape[0]

        compiled_model = self._compile_model()
        with self._profiler.stage_timer("capture.compiled_model"):
            output = compiled_model(**padded_kwargs)
        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer
        logits_buffer.copy_(output)
        elapsed = time.perf_counter() - t_start
        self._profiler.record_capture(elapsed, self._last_padded_batch or 0)

        return logits_buffer[:, :num_tokens]

    @record_function("ascend_piecewise_forward")
    def forward(self, **kwargs):
        """Replay captured graph."""
        if self._compile_miss_count and self._compiled_model is None:
            # shouldn't happen, but defensive
            logger.warning("GraphCaptureSession compile miss despite recorded misses.")
        if self._compiled_model is None:
            return self.capture(**kwargs)

        t_start = time.perf_counter()
        num_tokens = kwargs["input_ids"].size(-1)
        with self._profiler.stage_timer("forward.fill_buffers"):
            new_inputs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        padded_batch = context.block_offsets.shape[0]
        if (
            self._last_padded_batch is not None
            and padded_batch != self._last_padded_batch
        ):
            logger.warning(
                "GraphCaptureSession padded batch size changed %s → %s "
                "(is_decoding=%s)",
                self._last_padded_batch,
                padded_batch,
                self.meta.is_decoding,
            )
            self._last_padded_batch = padded_batch
        with self._profiler.stage_timer("forward.compiled_model"):
            compiled_out = self._compiled_model(**new_inputs)

        logits_buffer = self.meta.output_buffers["logits"]
        if compiled_out.data_ptr() != logits_buffer.data_ptr():
            logits_buffer.copy_(compiled_out)
        elapsed = time.perf_counter() - t_start
        self._profiler.record_replay(elapsed, padded_batch)
        return logits_buffer[:, :num_tokens]

    def _ensure_backend(self) -> None:
        if self._backend is None:
            self._backend = create_backend()
            if is_debug_enabled():
                logger.info(
                    "Created new backend for session (is_decoding=%s)",
                    self.meta.is_decoding,
                )

    def _compile_model(self):
        import torch._dynamo as dynamo

        if self._compiled_model is not None:
            return self._compiled_model

        cache_limit = dynamo.config.cache_size_limit
        if cache_limit < 1000:
            dynamo.config.cache_size_limit = 3000
            if is_debug_enabled():
                logger.info(
                    "Raised torch._dynamo cache_size_limit %s → %s for piecewise capture",
                    cache_limit,
                    dynamo.config.cache_size_limit,
                )

        self._compiled_model = torch.compile(
            self.model,
            backend=self._backend,
            fullgraph=True,
            dynamic=False,
        )
        self._compile_miss_count += 1
        return self._compiled_model

    def stats(self) -> Dict[str, Any]:
        """Return capture/replay statistics for debugging."""
        return {
            "capture_count": self._profiler.capture_count,
            "replay_count": self._profiler.replay_count,
            "compile_miss_count": self._compile_miss_count,
            "last_padded_batch": self._last_padded_batch,
            "is_decoding": self.meta.is_decoding,
            "capture_time_total_ms": self._profiler.capture_time_total * 1000,
            "replay_time_total_ms": self._profiler.replay_time_total * 1000,
            "stage_totals_ms": {
                name: total * 1000 for name, total in self._profiler.stage_totals.items()
            },
        }


__all__ = [
    "GraphCaptureSession",
    "AscendPiecewiseGraphMeta",
    "AscendPiecewiseAttentionBuffer",
    "make_buffers_cudagraph",
    "fill_buffers_cudagraph",
    "update_context_cudagraph",
]
