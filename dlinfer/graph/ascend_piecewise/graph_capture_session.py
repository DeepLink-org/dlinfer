"""Graph capture session for Ascend piecewise graph mode."""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from .utils import is_debug_enabled, PiecewiseEnvConfig
from .piecewise_backend import create_backend, get_ascend_compatible_size


logger = get_logger("dlinfer.ascend.capture")
if PiecewiseEnvConfig.get_debug_capture():
    logger.setLevel(logging.INFO)

BuffType = Dict[str, torch.Tensor]


class BufferFiller:
    """Buffer filling utilities."""

    @staticmethod
    def fill_token_buffers(
        input_buffers: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_tokens: int,
        padded_batch_size: int,
    ) -> None:
        """Fill token-related buffers."""
        input_ids_buf = input_buffers["input_ids"]
        # Handle 2D tensor copying for input_ids
        if input_ids.dim() == 2 and input_ids_buf.dim() == 2:
            input_ids_buf[0, :num_tokens].copy_(input_ids[0])
        else:
            # Simple copy and pad
            input_ids_buf[:1].copy_(input_ids)
            if 1 < padded_batch_size:
                input_ids_buf[1:padded_batch_size].zero_()

        # Zero remaining tokens if needed
        if num_tokens < padded_batch_size:
            input_ids_buf[0, num_tokens:padded_batch_size].zero_()

        position_ids_buf = input_buffers["position_ids"]
        # Handle 2D tensor copying for position_ids
        if position_ids.dim() == 2 and position_ids_buf.dim() == 2:
            position_ids_buf[0, :num_tokens].copy_(position_ids[0])
        else:
            # Simple copy and pad
            position_ids_buf[:1].copy_(position_ids)
            if 1 < padded_batch_size:
                position_ids_buf[1:padded_batch_size].zero_()

        # Zero remaining positions if needed
        if num_tokens < padded_batch_size:
            position_ids_buf[0, num_tokens:padded_batch_size].zero_()

    @staticmethod
    def fill_attention_metadata(
        input_buffers: Dict[str, Tensor],
        block_offsets: Tensor,
        kv_seqlens: Tensor,
        kv_start_indices: Tensor,
        batch_size: int,
        num_blocks: int,
        padded_batch_size: int,
    ) -> None:
        """Fill attention metadata buffers."""
        block_offsets_buf = input_buffers["block_offsets"]
        # Use precise slicing for 2D tensors
        block_offsets_buf[:batch_size, : block_offsets.size(1)].copy_(block_offsets)
        if batch_size < padded_batch_size:
            block_offsets_buf[batch_size:padded_batch_size, :num_blocks].zero_()

        kv_seqlens_buffer = input_buffers["kv_seqlens"]
        kv_seqlens_buffer[:batch_size].copy_(kv_seqlens.cpu())
        if batch_size < padded_batch_size:
            kv_seqlens_buffer[batch_size:padded_batch_size].zero_()

        kv_start_buffer = input_buffers["kv_start_indices"]
        kv_start_indices_tensor = kv_start_indices.to(
            device=kv_start_buffer.device, dtype=kv_start_buffer.dtype
        )
        kv_start_buffer[:batch_size].copy_(kv_start_indices_tensor)
        if batch_size < padded_batch_size:
            kv_start_buffer[batch_size:padded_batch_size].zero_()

    @staticmethod
    def fill_embeddings_buffer(
        input_buffers: Dict[str, Tensor],
        inputs_embeds: Tensor,
        num_tokens: int,
        padded_batch_size: int,
    ) -> None:
        """Fill embeddings buffer."""
        if inputs_embeds is None:
            return

        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )

        inputs_embed_buf = input_buffers["inputs_embeds"]
        # Handle 3D tensor copying for embeddings
        if inputs_embeds.dim() == 3 and inputs_embed_buf.dim() == 3:
            inputs_embed_buf[0, :num_tokens, :].copy_(inputs_embeds[0])
        else:
            # Simple copy and pad
            inputs_embed_buf[:1].copy_(inputs_embeds)
            if 1 < padded_batch_size:
                inputs_embed_buf[1:padded_batch_size].zero_()

        if num_tokens < padded_batch_size:
            inputs_embed_buf[0, num_tokens:padded_batch_size, :].zero_()

    @staticmethod
    def update_attention_metadata(
        attn_metadata: Any,
        input_buffers: Dict[str, Tensor],
        output_buffers: Dict[str, Tensor],
        padded_batch_size: int,
    ) -> None:
        """Update attention metadata references."""
        attn_metadata.block_offsets = input_buffers["block_offsets"][:padded_batch_size]
        attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:padded_batch_size]
        attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][
            :padded_batch_size
        ]
        attn_metadata.attn_output_buffer = output_buffers["attention_output"][
            :padded_batch_size
        ]


@dataclass
class AscendPiecewiseGraphMeta(CudaGraphMeta):
    """Metadata for piecewise graph optimization."""

    max_batchs: int  # Keep original name for lmdeploy compatibility
    max_tokens: int
    num_blocks: int
    is_decoding: int
    device: torch.device
    head_dim: int = 0
    num_attention_heads: int = 0
    dtype: torch.dtype = torch.float16

    @property
    def max_batches(self) -> int:
        """Backward compatible property accessor."""
        return self.max_batchs


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


def make_buffers_cudagraph(
    graph_meta: CudaGraphMeta, *args: Any, **kwargs: Any
) -> BuffType:
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


@record_function("ascend_piecewise_fill_buffers")
def fill_buffers_cudagraph(
    graph_meta: CudaGraphMeta,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: List,
    attn_metadata: Any,
    inputs_embeds: torch.Tensor,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Simplified buffer filling function using BufferFiller utilities."""
    # Extract metadata
    block_offsets: torch.Tensor = attn_metadata.block_offsets
    kv_seqlens: torch.Tensor = attn_metadata.kv_seqlens
    kv_start_indices: torch.Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers
    output_buffers: BuffType = graph_meta.output_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # Calculate padded batch size
    if graph_meta.is_decoding:
        padded_batch_size = max(graph_meta.max_batchs, batch_size)
    else:
        padded_batch_size = get_ascend_compatible_size(batch_size)

    # Use BufferFiller for buffer operations
    BufferFiller.fill_token_buffers(
        input_buffers, input_ids, position_ids, num_tokens, padded_batch_size
    )

    BufferFiller.fill_attention_metadata(
        input_buffers,
        block_offsets,
        kv_seqlens,
        kv_start_indices,
        batch_size,
        num_blocks,
        padded_batch_size,
    )

    BufferFiller.fill_embeddings_buffer(
        input_buffers, inputs_embeds, num_tokens, padded_batch_size
    )

    # Update attention metadata references
    BufferFiller.update_attention_metadata(
        attn_metadata, input_buffers, output_buffers, padded_batch_size
    )

    # Build new input dictionary
    new_inputs = {
        "past_key_values": past_key_values,
        "attn_metadata": attn_metadata,
        "input_ids": input_buffers["input_ids"][:, :padded_batch_size],
        "position_ids": input_buffers["position_ids"][:, :padded_batch_size],
    }

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][
            :, :padded_batch_size
        ]

    # Handle extra kwargs - currently only processing standard keys
    handled_keys = {
        "input_ids",
        "position_ids",
        "inputs_embeds",
        "attn_metadata",
        "past_key_values",
    }

    return new_inputs


@record_function("ascend_piecewise_update_context")
def update_context_cudagraph(graph_meta, context):
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]


class GraphCaptureSession:
    """Graph capture session."""

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
        self.device = device

        # State management
        self._compile_miss_count = 0
        self._last_padded_batch: Optional[int] = None

        # Graph metadata
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

        # Compilation state
        self._compiled_model = None
        self._backend = None

    @record_function("ascend_piecewise_capture")
    def capture(self, **kwargs):
        """Compile and capture graph for the first time."""
        t_start = time.perf_counter()
        num_tokens = kwargs["input_ids"].size(-1)

        # Initialize backend and buffers
        self._ensure_backend()
        self.meta.input_buffers, self.meta.output_buffers = make_buffers_cudagraph(
            self.meta, **kwargs
        )

        # Fill buffers
        padded_kwargs = fill_buffers_cudagraph(self.meta, **kwargs)

        # Debug info handling
        if PiecewiseEnvConfig.get_debug_capture():
            input_ids = padded_kwargs.get("input_ids")
            attn_metadata = padded_kwargs.get("attn_metadata")
            meta = getattr(attn_metadata, "kv_seqlens", None)
            debug_info = (
                f"padded input_ids={tuple(input_ids.shape) if input_ids is not None else None} "
                f"kv_seqlens={tuple(meta.shape) if meta is not None else None} "
                f"q_seqlens={getattr(attn_metadata, 'q_seqlens', None)}"
            )
            self._last_capture_debug_info = debug_info

        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        self._last_padded_batch = context.block_offsets.shape[0]

        # Compile and execute model
        compiled_model = self._compile_model()
        output = compiled_model(**padded_kwargs)

        # Handle output buffer
        logits_buffer = self._ensure_logits_buffer(output)
        logits_buffer.copy_(output)

        return logits_buffer[:, :num_tokens]

    def _ensure_logits_buffer(self, output: torch.Tensor) -> torch.Tensor:
        """Ensure logits buffer exists and has correct shape."""
        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer
        return logits_buffer

    @record_function("ascend_piecewise_forward")
    def forward(self, **kwargs):
        """Replay captured graph."""
        # Defensive checks
        if self._compile_miss_count and self._compiled_model is None:
            logger.warning("GraphCaptureSession compile miss despite recorded misses.")
        if self._compiled_model is None:
            return self.capture(**kwargs)

        t_start = time.perf_counter()
        num_tokens = kwargs["input_ids"].size(-1)

        # Fill buffers
        new_inputs = fill_buffers_cudagraph(self.meta, **kwargs)

        # Update context
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        padded_batch = context.block_offsets.shape[0]

        # Check batch size change
        self._check_batch_size_change(padded_batch)

        # Execute compiled model
        compiled_out = self._compiled_model(**new_inputs)

        # Handle output
        logits_buffer = self.meta.output_buffers["logits"]
        if compiled_out.data_ptr() != logits_buffer.data_ptr():
            logits_buffer.copy_(compiled_out)

        return logits_buffer[:, :num_tokens]

    def _check_batch_size_change(self, padded_batch: int) -> None:
        """Check and log batch size changes."""
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
            "compile_miss_count": self._compile_miss_count,
            "last_padded_batch": self._last_padded_batch,
            "is_decoding": self.meta.is_decoding,
        }


__all__ = [
    "GraphCaptureSession",
    "AscendPiecewiseGraphMeta",
    "AscendPiecewiseAttentionBuffer",
]
