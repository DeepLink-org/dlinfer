# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
import torch
from torch import Tensor

from typing import Any, Dict, List

from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin, next_power_of_2

BuffType = Dict[str, Tensor]

'''
this file implements the cudagraph for ascend backend.
'''

'''
Ascend CudaGraphMixin methods
for cudagraph buffer management.
'''

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

    input_buffers["kv_seqlens"] = torch.zeros(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches, 1), dtype=torch.int64, device=device
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
    **kwargs
) -> Dict[str, Tensor]:
    """fill cudagraph buffers from forward inputs."""
    block_offsets: Tensor = attn_metadata.block_offsets
    q_start_loc: Tensor = attn_metadata.q_start_loc
    q_seqlens: Tensor = attn_metadata.q_seqlens
    kv_seqlens: Tensor = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    input_buffers["q_start_loc"][: batch_size + 1] = q_start_loc
    input_buffers["q_seqlens"][:batch_size] = q_seqlens
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds
    # create inputs
    new_batch_size = next_power_of_2(batch_size)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.q_start_loc = input_buffers["q_start_loc"][: new_batch_size + 1]
    attn_metadata.q_seqlens = input_buffers["q_seqlens"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
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


CudaGraphMixin.make_buffers_cudagraph = AscendCudaGraphMixin_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph = AscendCudaGraphMixin_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = AscendCudaGraphMixin_update_context_cudagraph


"""
implement Ascend CudaGraphRunner and CUDASingleGraphRunner. 
"""

import functools
from typing import Any, Dict, List, Tuple

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner

logger = get_logger('lmdeploy')


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


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size."""
    ret = []
    batch_size = 1
    batch_step = 256
    # power of 2
    while batch_size <= min(batch_step, max_batches):
        ret.append(batch_size)
        batch_size *= 2

    # step
    ret += list(range(batch_size, max_batches + 1, batch_step))

    if max_batches != ret[-1]:
        ret.append(max_batches)
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
        pool: Tuple[int, int],
        model_config: ModelConfig,
        device: torch.device,
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
        self._graph: torch.cuda.CUDAGraph = None

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        self.model(**padded_kwargs)

        self._graph = torch.cuda.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.cuda.graph(self._graph, pool=self.pool, stream=current_stream, capture_error_mode='thread_local'):
            output = self.model(**padded_kwargs)

        output_buffers = dict(logits=output)
        self.meta.output_buffers = output_buffers
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs['input_ids'].size(-1)
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        self._graph.replay()

        output = self.meta.output_buffers['logits'][:, :num_tokens]
        return output

    def __del__(self):
        """del."""
        del self._graph


class CUDAGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, CUDASingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _try_compile_model_once(self):
        if self.has_try_compile_model:
            return

        # TODO: recovery it when torch.compile is stable (should be add a flag to enable it?)
        # if hasattr(self.model, 'compile_model'):
        #     method = getattr(self.model, 'compile_model')
        #     method()

        self.has_try_compile_model = True

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: Any, inputs_embeds: torch.Tensor, **kwargs):
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
        if not self.backend_config.eager_mode and get_backend().get_name() == 'cuda':
            self._try_compile_model_once()

        enable_graph = self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function('forward_eager'):
                return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        if graph_key not in self._runner_map:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = CUDASingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=max_tokens,
                                           num_blocks=self.num_blocks,
                                           is_decoding=is_decoding,
                                           pool=self.graph_pool_handle,
                                           model_config=self.model_config,
                                           device=self.device)
            runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
        else:
            runner = self._runner_map[graph_key]
        output = runner.forward(**kwargs)
        return output

    @record_function('prepare_inputs_for_generation')
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
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
