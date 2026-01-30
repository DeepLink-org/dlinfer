# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
# this file implements the cudagraph for ascend backend.
import functools
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from contextlib import ExitStack
from packaging.version import InvalidVersion, Version

import torch
import torch_npu
from torch import Tensor
from torch.profiler import record_function

from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.backends.graph_runner import GraphRunner

from lmdeploy.utils import get_logger

logger = get_logger("dlinfer")
BuffType = Dict[str, Tensor]


@functools.lru_cache()
def aclgraph_use_torch_npu_update():
    min_valid_version = Version("2.8.0.post1")

    try:
        current_version = Version(torch_npu.__version__)
    except InvalidVersion:
        return False

    return current_version >= min_valid_version


# AscendCudaGraphMixin methods for cudagraph buffer management.
def AscendCudaGraphMixin_make_buffers_cudagraph(
    self, graph_meta: CudaGraphMeta, *args, **kwargs
) -> BuffType:
    """make cudagraph buffers from forward inputs."""
    max_batches = graph_meta.max_batchs
    max_tokens = graph_meta.max_tokens
    num_blocks = graph_meta.num_blocks
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

    input_buffers["q_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["kv_seqlens"] = torch.ones(max_batches, dtype=torch.int32)

    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches), dtype=torch.int32, device=device
    )

    input_buffers["x_active_mask"] = torch.zeros(
        (max_batches), dtype=torch.bool, device=device
    )
    return input_buffers


def AscendCudaGraphMixin_fill_buffers_cudagraph(
    self,
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
    kv_seqlens: Tensor = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices
    moe_metadata = get_step_ctx_manager().current_context().moe_metadata
    x_active_mask: Tensor = moe_metadata.x_active_mask
    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices
    if x_active_mask is not None:
        input_buffers["x_active_mask"][:batch_size] = x_active_mask

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds
    # create inputs
    # Use compatible size but cap at graph's max_batchs to avoid buffer overflow
    new_batch_size = min(get_ascend_compatible_size(batch_size), graph_meta.max_batchs)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]
    moe_metadata.x_active_mask = input_buffers["x_active_mask"][:new_batch_size]

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
        moe_metadata=moe_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"][:, :new_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :new_batch_size]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][:, :new_batch_size]

    new_inputs.update(kwargs)

    return new_inputs


def AscendCudaGraphMixin_update_context_cudagraph(self, graph_meta, context):
    """update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]
    context.moe_metadata.x_active_mask = input_buffers["x_active_mask"]


CudaGraphMixin.make_buffers_cudagraph = AscendCudaGraphMixin_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph = AscendCudaGraphMixin_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = AscendCudaGraphMixin_update_context_cudagraph


def next_power_of_2(n: int):
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


def get_ascend_compatible_size(n: int):
    """Get ascend compatible size."""
    if n <= 16:
        n = next_power_of_2(n)
    elif n <= 256:
        n = (n + 15) & ~0xF
    else:
        n = (((n - 1) >> 8) + 1) << 8
    return n


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size.

    Generate compatible sizes up to max_batches (not exceeding it),
    then add max_batches itself to ensure it can be handled.
    """
    ret = []
    batch_size = 1

    # Generate batch sizes and apply get_ascend_compatible_size
    # Only include sizes that do not exceed max_batches
    while batch_size <= max_batches:
        compatible_size = get_ascend_compatible_size(batch_size)
        if compatible_size > max_batches:
            break
        if not ret or compatible_size > ret[-1]:
            ret.append(compatible_size)
        batch_size = compatible_size + 1

    # Add max_batches itself to ensure it can be handled
    if max_batches not in ret:
        ret.append(max_batches)

    set_graph_params(set(ret))
    return ret


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class AscendSingleGraphRunner:
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
        update_stream: torch.npu.Stream,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.npu.NPUGraph = None
        self.update_stream = update_stream

    @record_function("capture_cudagraph")
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f"Capturing graph with meta: {self.meta}")
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        aclgraph = torch.npu.NPUGraph()
        with ExitStack() as stack:
            with torch.npu.graph(
                aclgraph,
                auto_dispatch_capture=True,
                pool=self.pool,
                stream=current_stream,
            ):
                output = self.model(**padded_kwargs)

        output_buffers = dict(logits=output)
        self.meta.output_buffers = output_buffers
        self._graph = aclgraph
        return output

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs["input_ids"].size(-1)
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        if aclgraph_use_torch_npu_update():
            self._graph.replay()
            self._graph.update(
                cpu_update_input=[
                    {"actual_seq_lengths_kv": self.meta.input_buffers["kv_seqlens"]}
                ]
            )
        else:
            update_attn_params(self.update_stream, self.meta, self.max_tokens)
            self._graph.replay()
        output = self.meta.output_buffers["logits"][:, :num_tokens]
        return output

    def reset(self):
        """Reset NPU graph and release its buffers."""
        if self._graph is not None:
            try:
                if hasattr(self._graph, "reset"):
                    self._graph.reset()
            finally:
                self._graph = None

        if hasattr(self.meta, "output_buffers") and isinstance(
            self.meta.output_buffers, dict
        ):
            self.meta.output_buffers.clear()

    def __del__(self):
        """Best-effort cleanup when runner is GC-ed."""
        try:
            self.reset()
        except Exception:
            pass


class AscendGraphRunner(GraphRunner):
    """Cuda graph runner."""

    capturing = False

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
        self._runner_map: Dict[Any, AscendSingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False
        self.update_stream = torch.npu.Stream()

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
        enable_graph = self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function("forward_eager"):
                ret = self.model(**kwargs)
                return ret

        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        if graph_key not in self._runner_map:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = AscendSingleGraphRunner(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
                update_stream=self.update_stream,
            )
            AscendGraphRunner.capturing = True
            runner.capture(**kwargs)
            AscendGraphRunner.capturing = False
            self._runner_map[graph_key] = runner
        else:
            runner = self._runner_map[graph_key]
        output = runner.forward(**kwargs)
        return output

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
        """Remove all graphs and related resources to prevent hanging on exit."""
        for _, runner in self._runner_map.items():
            try:
                runner.reset()
            except Exception as e:
                logger.warning(f"AscendGraphRunner.reset: runner.reset error: {e!r}")
        self._runner_map.clear()
        clear_graph_params()
        self.graph_pool_handle = None
        torch.npu.empty_cache()

    def __del__(self):
        """Best-effort cleanup when graph runner is GC-ed."""
        try:
            self.reset()
        except Exception:
            pass

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
        return _get_capture_batch_size_impl(self.cache_config.max_batches)


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    is_mla: bool


_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: set[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        False,
    )


def get_graph_params():
    return _graph_params


def clear_graph_params():
    """Clear global graph params and release references to KV cache tensors."""
    global _graph_params
    if _graph_params is None:
        return

    try:
        for k in list(_graph_params.attn_params.keys()):
            _graph_params.attn_params[k].clear()
        for k in list(_graph_params.handles.keys()):
            _graph_params.handles[k].clear()
        for k in list(_graph_params.events.keys()):
            _graph_params.events[k].clear()
        _graph_params.is_mla = None

        _graph_params.workspaces.clear()
    finally:
        _graph_params = None


def update_attn_params(update_stream, forward_meta, runtime_size):
    graph_params = get_graph_params()
    for param, handle, event in zip(
        graph_params.attn_params[runtime_size],
        graph_params.handles[runtime_size],
        graph_params.events[runtime_size],
    ):
        if graph_params.is_mla:
            update_decode_attention_mla_params(
                update_stream, forward_meta, param, handle, event
            )
        else:
            update_decode_attention_params(
                update_stream, forward_meta, param, handle, event
            )


def update_decode_attention_params(update_stream, forward_meta, param, handle, event):
    (
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale,
        block_table,
        kv_seq_len,
        output,
    ) = param
    kv_seq_len = forward_meta.input_buffers["kv_seqlens"]
    with torch.npu.stream(update_stream):
        torch.npu.graph_task_update_begin(update_stream, handle)
        torch.ops.atb._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=kv_seq_len,
            out=output,
        )
        torch.npu.graph_task_update_end(update_stream)
        event.record(update_stream)


def update_decode_attention_mla_params(
    update_stream, forward_meta, param, handle, event
):
    (
        query,
        key_cache,
        num_kv_heads,
        num_q_heads,
        scale_value,
        block_table,
        kv_seq_len,
        mla_vheadsize,
        attn_output,
    ) = param
    kv_seq_len = forward_meta.input_buffers["kv_seqlens"]
    with torch.npu.stream(update_stream):
        torch.npu.graph_task_update_begin(update_stream, handle)
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=mla_vheadsize,
            out=attn_output,
        )
        torch.npu.graph_task_update_end(update_stream)
        event.record(update_stream)
