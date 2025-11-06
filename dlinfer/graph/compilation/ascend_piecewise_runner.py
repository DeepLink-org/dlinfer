"""
Piecewise Graph Runner
与lmdeploy的warmup机制集成
"""
import torch
from torch import Tensor
from typing import Any, Dict, List, Tuple, Optional
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from dlinfer.graph.compilation.piecewise_backend import (
    create_backend,
    get_ascend_compatible_size,
    get_capture_batch_sizes as backend_capture_batch_sizes,
)
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager


from torch.profiler import record_function

logger = get_logger('lmdeploy')
logger = get_logger("dlinfer")
BuffType = Dict[str, Tensor]

# AscendCudaGraphMixin methods for cudagraph buffer management.
def make_buffers_cudagraph(
    graph_meta: CudaGraphMeta, *args, **kwargs
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

    input_buffers["kv_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches), dtype=torch.int64, device=device
    )
    return input_buffers


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
    # 说明：input_buffers 在 capture 阶段构建，并在整个 decode 生命周期中复用。
    # AscendPiecewiseGraphWrapper 依赖这些缓冲区的地址不变来保证 ACL graph 可重复执行。

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    # import pdb;pdb.set_trace()
    kv_seqlens_tensor = torch.as_tensor(
        kv_seqlens, dtype=torch.int32, device=input_buffers["kv_seqlens"].device
    )
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens_tensor
    kv_start_indices_tensor = torch.as_tensor(
        kv_start_indices,
        dtype=input_buffers["kv_start_indices"].dtype,
        device=input_buffers["kv_start_indices"].device,
    )
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices_tensor

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds
    # create inputs
    new_batch_size = get_ascend_compatible_size(batch_size)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"][:, :new_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :new_batch_size]
    # 说明：上述切片返回 view，与 input_buffers 共享底层存储，确保 graph capture/replay 地址一致。

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][:, :new_batch_size]

    new_inputs.update(kwargs)

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
        self.compiled_model = None
        self.backend = None  # 延迟初始化，确保同一个 runner 复用 backend

    @record_function("capture_cudagraph")
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f"Capturing graph with meta: {self.meta}")
        
        # 延迟初始化 backend，确保同一个 runner 复用
        if self.backend is None:
            self.backend = create_backend()
            logger.info(f"Created new backend for runner (is_decoding={self.is_decoding})")
        
        # Capture 阶段初始化持久化缓冲区；后续 replay 将反复复用同一块内存。
        self.meta.input_buffers = make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)

        # 优化torch.compile配置，减少编译开销
        import torch._dynamo
        original_cache_size = torch._dynamo.config.cache_size_limit
        if original_cache_size < 1000:
            torch._dynamo.config.cache_size_limit = 1000
            logger.info(
                "Raised torch._dynamo cache_size_limit %s → %s for piecewise capture",
                original_cache_size,
                torch._dynamo.config.cache_size_limit,
            )
        else:
            logger.info(
                "Using existing torch._dynamo cache_size_limit=%s for piecewise capture",
                original_cache_size,
            )

        self.compiled_model = torch.compile(
            self.model,
            backend=self.backend,
            fullgraph=True,  # 可以追踪完整图（custom ops 已注册）
            dynamic=False,  # 每个batch size单独编译一次
        )

        output = self.compiled_model(**padded_kwargs)

        # 预分配固定输出缓冲区，保持地址稳定
        if self.meta.output_buffers is None:
            self.meta.output_buffers = {}

        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer

        logits_buffer.copy_(output)
        return logits_buffer


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

        import dlinfer.graph
        dlinfer.graph.config.enable_graph_mode = True

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
            runner.capture(**kwargs)
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
