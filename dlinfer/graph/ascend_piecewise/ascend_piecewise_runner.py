"""
Piecewise Graph Runner
lmdeploywarmup
"""

from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Any, Dict, List, Tuple, Optional
from collections.abc import Mapping, Sequence
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
import dlinfer.graph
from dlinfer.graph.ascend_piecewise.piecewise_backend import (
    create_backend,
    get_ascend_compatible_size,
    get_capture_batch_sizes as backend_capture_batch_sizes,
)
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager

from torch.profiler import record_function

logger = get_logger("dlinfer")


def is_debug_enabled() -> bool:
    """Check if ACL graph debugging is enabled via environment variable."""
    import os

    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


BuffType = Dict[str, Tensor]


@dataclass
class AscendPiecewiseGraphMeta:
    """Metadata for piecewise graph optimization."""

    max_batchs: int
    max_tokens: int
    num_blocks: int
    is_decoding: int
    device: torch.device
    head_dim: int
    num_attention_heads: int
    dtype: torch.dtype
    input_buffers: BuffType = None
    output_buffers: BuffType = None
    vocab_size: int = 1


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
                logger.info(f"get_attention_output: tp={tp}, tp_rank={tp_rank}")
            cls.class_attention_output = torch.empty(
                batch_size,
                num_attention_heads // tp,
                head_dim,
                dtype=dtype,
                device=device,
            )
            return cls.class_attention_output
        else:
            return cls.class_attention_output[:batch_size]


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
    """Allocate (or reuse) a persistent buffer and copy tensor data into it."""

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
    """fill cudagraph buffers from forward inputs."""
    block_offsets: Tensor = attn_metadata.block_offsets
    kv_seqlens: List = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers
    output_buffers: BuffType = graph_meta.output_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"].zero_()
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"].zero_()
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets

    kv_seqlens_tensor = torch.as_tensor(
        kv_seqlens,
        dtype=torch.int32,
    )
    # device=input_buffers["kv_seqlens"].device
    input_buffers["kv_seqlens"].zero_()
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens_tensor

    kv_start_indices_tensor = torch.as_tensor(
        kv_start_indices,
        dtype=input_buffers["kv_start_indices"].dtype,
        device=input_buffers["kv_start_indices"].device,
    )
    input_buffers["kv_start_indices"].zero_()
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices_tensor

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        else:
            input_buffers["inputs_embeds"].zero_()
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds

    new_batch_size = get_ascend_compatible_size(batch_size)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]
    attn_metadata.attn_output_buffer = output_buffers["attention_output"][
        :new_batch_size
    ]

    q_seqlens_tensor = getattr(attn_metadata, "q_seqlens", None)
    if q_seqlens_tensor is not None:
        attn_metadata.q_seqlens = _ensure_tensor_view(
            graph_meta,
            "q_seqlens",
            q_seqlens_tensor,
            target_first_dim=new_batch_size,
        )

    q_start_loc_tensor = getattr(attn_metadata, "q_start_loc", None)
    if q_start_loc_tensor is not None:
        pad_dim = (
            new_batch_size
            if q_start_loc_tensor.dim() == 0
            else max(new_batch_size + 1, q_start_loc_tensor.shape[0])
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
            target_first_dim=new_batch_size,
        )

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"][:, :new_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :new_batch_size]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][:, :new_batch_size]

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
            pad_dim=new_batch_size,
        )
        extra_kwargs[key] = materialized

    new_inputs.update(extra_kwargs)

    return new_inputs


def update_context_cudagraph(graph_meta, context):
    """update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class AscendPiecewiseSingleGraphRunner:
    """Cuda single graph runner."""

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
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config

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
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.compiled_model = None
        self.backend = None

    @record_function("capture_cudagraph")
    def capture(self, **kwargs):
        """Capture graph."""
        if is_debug_enabled():
            logger.debug(f"Capturing graph with meta: {self.meta}")

        num_tokens = kwargs["input_ids"].size(-1)

        if self.backend is None:
            self.backend = create_backend()
            if is_debug_enabled():
                logger.info(
                    f"Created new backend for runner (is_decoding={self.is_decoding})"
                )

        self.meta.input_buffers, self.meta.output_buffers = make_buffers_cudagraph(
            self.meta, **kwargs
        )
        padded_kwargs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)

        import torch._dynamo as dynamo

        cache_limit = dynamo.config.cache_size_limit
        if cache_limit < 1000:
            dynamo.config.cache_size_limit = 1000
            if is_debug_enabled():
                logger.info(
                    "Raised torch._dynamo cache_size_limit %s → %s for piecewise capture",
                    cache_limit,
                    dynamo.config.cache_size_limit,
                )

        self.compiled_model = torch.compile(
            self.model,
            backend=self.backend,
            fullgraph=True,
            dynamic=False,
        )

        output = self.compiled_model(**padded_kwargs)

        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer

        logits_buffer.copy_(output)
        return logits_buffer[:, :num_tokens]

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs["input_ids"].size(-1)
        assert self.compiled_model is not None
        new_inputs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        compiled_out = self.compiled_model(**new_inputs)

        logits_buffer = self.meta.output_buffers["logits"]

        if compiled_out.data_ptr() != logits_buffer.data_ptr():
            logits_buffer.copy_(compiled_out)

        return logits_buffer[:, :num_tokens]

    def __del__(self):
        """del."""
        if self.compiled_model:
            del self.compiled_model


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
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks
        self.enable_graph = self.check_enable_graph()
        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, AscendPiecewiseSingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

        dlinfer.graph.config.enable_graph_mode = True
        dlinfer.graph.config.piecewise_graph_enabled = True

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, "support_cuda_graph", _false)

    def _get_capture_tokens(self, batch_size: int):
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
    ):
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

    def __call__(self, **kwargs):
        """call."""
        if not self.enable_graph(**kwargs):
            with record_function("forward_eager"):
                return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        runner = self._runner_map.get(graph_key)
        if runner is None:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = AscendPiecewiseSingleGraphRunner(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
            )
            self._runner_map[graph_key] = runner

            from dlinfer.graph import config

            original_is_capturing = config.is_capturing
            config.is_capturing = True

            try:
                return runner.capture(**kwargs)
            finally:
                config.is_capturing = original_is_capturing

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

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        self._runner_map.clear()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            tp_size = self._get_capture_tokens(padding_batch_size)
            dp_meta.tp_sizes = [tp_size] * len(dp_meta.tp_sizes)
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        return backend_capture_batch_sizes(self.cache_config.max_batches)
