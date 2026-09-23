# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
# this file implements the cudagraph for ascend backend.
import functools
import os
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
    # V4_FORCE_ACL_UPDATE_PATH: override which per-step graph-update path is
    # used. The torch_npu>=2.8 path (replay -> _graph.update) orders the
    # actual_seq_lengths_kv CPU-param write AFTER replay with no sync, which
    # races the next replay's read (507018, see vllm-ascend acl_graph.py:243).
    # The legacy path (update_attn_params on a separate update_stream -> replay)
    # mirrors vllm-ascend's stable update-before-replay ordering. =0 forces the
    # legacy path to test whether the torch_npu_update ordering is the racer.
    if os.environ.get('V4_FORCE_ACL_UPDATE_PATH', '') == '0':
        return False
    if os.environ.get('V4_FORCE_ACL_UPDATE_PATH', '') == '1':
        return True
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
    input_buffers["input_ids"] = torch.zeros(
        (1, max_tokens), dtype=torch.int32, device=device
    )

    input_buffers["position_ids"] = torch.zeros(
        (1, max_tokens), dtype=torch.int32, device=device
    )

    input_buffers["block_offsets"] = torch.zeros(
        (max_batches, num_blocks), dtype=torch.int32, device=device
    )

    input_buffers["q_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    # V4_FULL_GRAPH_DECODE: V4 computes its decode metadata (slot_mapping /
    # cos-gather / start_pos) IN-GRAPH from the kv_seqlens graph input. That
    # requires kv_seqlens to be a DEVICE buffer (in-graph ops can't read CPU).
    # The default CPU buffer backs the standard op's actual_seq_lengths_kv CPU
    # param + the vestigial _graph.update() -- both UNUSED by V4 (V4 uses
    # sparse_attn_sharedkv with a device seqused_kv, and FULL-graph implies
    # V4_SKIP_ACL_UPDATE since a device tensor is incompatible with the CPU
    # actual_seq_lengths_kv update param). Non-V4 / non-FULL-graph keeps CPU.
    _kv_dev = (device if os.environ.get('V4_FULL_GRAPH_DECODE', '0') == '1'
               else None)
    input_buffers["kv_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=_kv_dev)

    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches), dtype=torch.int32, device=device
    )

    input_buffers["x_active_mask"] = torch.zeros(
        (max_batches), dtype=torch.bool, device=device
    )

    # ssm
    if graph_meta.is_ssm:
        input_buffers["state_ids"] = torch.full(
            (max_batches,), -1, dtype=torch.int64, device=device
        )

    # mrope
    if graph_meta.use_mrope:
        input_buffers["mrope_position_ids"] = torch.zeros(
            3, max_tokens, dtype=torch.int64, device=device
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
    q_start_loc: Tensor = attn_metadata.q_start_loc

    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    max_num_tokens = input_buffers["input_ids"].size(-1)
    if num_tokens < max_num_tokens:
        # Only initialize the padding region to avoid unnecessary work and RNG advancement
        input_buffers["input_ids"][:, num_tokens:max_num_tokens].random_(
            0, graph_meta.vocab_size
        )
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"].zero_()
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"].zero_()
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    input_buffers["kv_seqlens"].fill_(0)
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens
    input_buffers["kv_start_indices"].fill_(-1)
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices
    if x_active_mask is not None:
        input_buffers["x_active_mask"].fill_(0)
        input_buffers["x_active_mask"][:batch_size] = x_active_mask

    # ssm
    if graph_meta.is_ssm:
        input_buffers["q_start_loc"][: batch_size + 1] = q_start_loc
        input_buffers["q_start_loc"][batch_size + 1 :] = q_start_loc[-1]

        state_ids = kwargs["state_ids"]
        input_buffers["state_ids"].fill_(-1)
        input_buffers["state_ids"][: state_ids.size(0)].copy_(state_ids)

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

    attn_metadata.block_offsets = input_buffers["block_offsets"]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"]
    moe_metadata.x_active_mask = input_buffers["x_active_mask"]
    attn_metadata.q_start_loc = input_buffers["q_start_loc"]

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
        moe_metadata=moe_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"]
    new_inputs["position_ids"] = input_buffers["position_ids"]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"]

    new_inputs.update(kwargs)

    # ssm: override kwargs' variable-length state_ids with the fixed-size buffer
    if graph_meta.is_ssm:
        new_inputs["state_ids"] = input_buffers["state_ids"]

    # mrope
    if graph_meta.use_mrope:
        mrope_position_ids = kwargs.get("mrope_position_ids", None)
        if mrope_position_ids is not None:
            input_buffers["mrope_position_ids"].zero_()
            input_buffers["mrope_position_ids"][:, :num_tokens] = mrope_position_ids
        new_inputs["mrope_position_ids"] = input_buffers["mrope_position_ids"]

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

    # ssm
    if graph_meta.is_ssm:
        context.state_offsets = input_buffers["state_ids"]

    # mrope
    if graph_meta.use_mrope:
        context.mrope_position_ids = input_buffers["mrope_position_ids"]


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
            is_ssm=len(model_config.states_shapes) > 0,
            use_mrope=model_config.use_mrope,
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
        logger.debug(f"Capturing decode graph: is_decoding={self.is_decoding} "
                     f"max_batches={self.max_batches} max_tokens={self.max_tokens}")
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        warmup_output = self.model(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        aclgraph = torch.npu.NPUGraph()
        with ExitStack() as stack:
            AscendGraphRunner.capturing = True
            with torch.npu.graph(
                aclgraph,
                auto_dispatch_capture=True,
                pool=self.pool,
                stream=current_stream,
            ):
                graph_output = self.model(**padded_kwargs)
            AscendGraphRunner.capturing = False

        output_buffers = self.model.make_output_buffers(graph_output)
        self.meta.output_buffers = output_buffers
        self._graph = aclgraph
        # Latch: a graph is now captured -> subsequent build_context calls run
        # on the replay path (device kv_seqlens), so v4_dsa elides the eager
        # metadata (None sentinels) and the model forward computes it in-graph.
        AscendGraphRunner.has_captured = True
        final_output = self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)
        return final_output

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """forward."""
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        if aclgraph_use_torch_npu_update():
            # Drain the current stream before replay (mirrors vllm-ascend
            # acl_graph.py:199) so the previous step's replay + any per-step
            # eager aclnn op queued on the default stream has fully retired
            # before this replay launches -> correct replay ordering.
            # NOTE: torch.npu.synchronize() (device-wide) was tried and is
            # WORSE -- it drains the HCCL internal stream too and so surfaces
            # the residual in-graph vector/HCCL racer (507035, "cross-device
            # memory access timeout" on the EP rank) EARLIER (req9 vs req53).
            # A sync is a DETECTOR of that racer, not a fixer; the real fix is
            # the FULL-graph plan (move lm_head/sampling/build_step_context
            # in-graph -> 0 eager ops between replays, mirrors vllm stability).
            # V4_DEVICE_SYNC=1: switch to torch.npu.synchronize() (device-wide,
            # drains the HCCL internal stream too). This does NOT fix the racer
            # -- it is a DETECTOR that surfaces the in-graph vector/HCCL racer
            # (507035, "cross-device memory access timeout" on the EP rank)
            # EARLIER (req9 vs q56), turning the 4h non-deterministic GPQA
            # crash into a ~30min deterministic repro for fix iteration. The
            # real fix remains the FULL-graph plan (0 eager ops between replays
            # / proper HCCL cross-stream sync). Default 0 = current-stream sync.
            if os.environ.get('V4_DEVICE_SYNC', '0') == '1':
                torch.npu.synchronize()
            else:
                torch.npu.current_stream().synchronize()
            self._graph.replay()
            # V4_SKIP_ACL_UPDATE: V4's attention op is _C_ascend.
            # sparse_attn_sharedkv, which reads kv length from the
            # seqused_kv DEVICE tensor (refreshed per-step via
            # fill_buffers_cudagraph into the pinned buffer, read at stable
            # address at replay) -- NOT from the standard op's
            # actual_seq_lengths_kv CPU param. V4's captured graph therefore
            # contains NO actual_seq_lengths_kv CPU param, so this
            # _graph.update() is VESTIGIAL for V4 (mirrors vllm-ascend
            # dsa_v1.py update_graph_params = pass no-op). It is a per-step
            # mutation of graph internal state for a param the graph does not
            # have, which (while a previous replay may still be in flight) is
            # a candidate racer for the 507018 timing race. =1 skips it for V4
            # (decode only); non-V4 models keep the update (they DO use the
            # standard op with the CPU param).
            if os.environ.get('V4_SKIP_ACL_UPDATE', '0') != '1' and \
                    os.environ.get('V4_FULL_GRAPH_DECODE', '0') != '1':
                self._graph.update(
                    cpu_update_input=[
                        {"actual_seq_lengths_kv": self.meta.input_buffers["kv_seqlens"]}
                    ]
                )
        else:
            update_attn_params(self.update_stream, self.meta, self.max_tokens)
            self._graph.replay()
        output_buffers = self.meta.output_buffers
        output = self.model.get_outputs_cudagraph(output_buffers, **kwargs)
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
    # Class-level "a graph has been captured" latch. Set True at the end of
    # AscendSingleGraphRunner.capture() (after self._graph is assigned), so it
    # is False during the capture build_context / warmup and True from the
    # first replay onward. Read by v4_dsa.build_v4_dsa_inputs to gate the
    # FULL-graph (V4_FULL_GRAPH_DECODE) eager-metadata elision: the SHARED
    # decode metadata (cos gather / slot_mapping / start_pos / _pin_meta
    # copies) is computed eagerly only when NO graph is captured yet (capture
    # build_context + pre-capture warmup, where attn_metadata.kv_seqlens is
    # CPU) and elided (None sentinels, computed IN-GRAPH by the model forward)
    # once a graph is captured (replay path, where kv_seqlens is a device
    # graph-input buffer). This yields 0 eager aclnn metadata racers between
    # replays (mirrors vllm-ascend FULL graph) -> stable (fixes 507018).
    has_captured = False

    # Option A: per-step "route this decode step eager" flag, set by
    # v4_dsa.build_v4_dsa_inputs (True on a batch-change / callonce-miss
    # decode step where a kv_seqlens.tolist() sync would race the replay).
    # Read in __call__'s replay branch. Default False (steady / capture /
    # prefill). Class attr: one AscendGraphRunner per worker process, and
    # build_v4_dsa_inputs runs in build_context (eager, pre-__call__) on
    # the same worker, so the attr is fresh when __call__ reads it.
    _v4_step_route_eager = False

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
                return self.model.make_output_buffers(ret)

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
            runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
        else:
            runner = self._runner_map[graph_key]
        # Option A (batch-change -> eager): a decode REPLAY step whose batch
        # composition changed (num_reqs up/down, a slot freed+reused) misses
        # the _slot_kv tracker -> build_v4_dsa_inputs falls back to the
        # kv_seqlens.tolist() D2H host sync, which races the in-flight
        # captured-graph HCCL collective (507035 "vector core abnormal").
        # (callonce cache misses, _meta_all_hit False, race the same way.)
        # Route such steps EAGER: the syncs run where no captured replay can
        # race them (documented design intent, v4_dsa.py:867-881), and the
        # caches re-seed so the next steady step replays cleanly. The
        # _v4_step_route_eager flag is computed in build_v4_dsa_inputs
        # (build_context), which runs eager PRE-__call__ on this worker, so
        # it is fresh here. Identical inputs across ranks -> identical flag
        # -> rank-uniform decision (no EP-collective graph-entry mismatch,
        # vllm-ascend "min-mode"). Decode-only + replay-branch only; capture
        # (graph_key absent) is unaffected. Toggled by V4_BATCH_CHANGE_EAGER.
        if (is_decoding
                and os.environ.get('V4_BATCH_CHANGE_EAGER', '1') == '1'
                and getattr(self, '_v4_step_route_eager', False)):
            with record_function("forward_eager_batch_change"):
                if os.environ.get('V4_BC_EAGER_DEBUG', '0') == '1':
                    import torch.distributed as _d
                    _rk = (_d.get_rank() if (_d.is_available()
                                             and _d.is_initialized()) else 0)
                    print(f'[V4-BCE] rank={_rk} graph_key={graph_key} '
                          f'batch-change -> eager (no replay race)',
                          flush=True)
                ret = self.model(**kwargs)
                return self.model.make_output_buffers(ret)
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
        dp_meta = inputs.dp_meta
        # Key the decode-graph tp_sizes update off the GLOBAL decode flag
        # (dp_meta.dp_is_decoding, set from the gathered global_is_decoding
        # in agent._prepare_dp_v1), not the per-rank inputs.is_decoding.
        # Under DP a step can be mixed -- e.g. one DP group prefilling while
        # another carries a dummy decode -- and the local flag on the
        # dummy-decode group is True even though the whole step is a prefill
        # (global_is_decoding=False). The local flag would then drive the
        # decode-graph branch and read an unset padding_batch_size (None).
        # The global flag is consistent with whether padding_batch_size was
        # set (both derive from global_is_decoding), so use it.
        if dp_meta is not None and getattr(dp_meta, 'dp_is_decoding', False):
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            tp_size = self._get_capture_tokens(padding_batch_size)
            dp_meta.tp_sizes = [tp_size] * len(dp_meta.tp_sizes)
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        # Capture graphs for the FULL engine max_batches (not a DP-halved
        # per-rank ceiling). An earlier optimization assumed internal DP always
        # splits the running batch evenly (ceil(max_batches/dp) per rank) and
        # capped the capture set there. That assumption is FALSE: the DP
        # scheduler can route more than ceil(max_batches/dp) requests to one
        # group (e.g. max_batches=2/dp=2 -> 2 concurrent decodes on the same
        # group while the other runs a dummy). The DP-halved capture set
        # (=[1]) then has no graph for a batch=2 decode -> AssertionError
        # "Unsupported batch_size=2" (and the V4 state pool, likewise
        # DP-halved, IndexErrors first) -> whole-serve ENGINE_STOP crash.
        # Capturing the full max_batches (=[1,2] for mb=2) makes the graph
        # robust to any DP routing, and matches the (likewise full-sized)
        # V4 state pool so capture-warmup state_bt ranges stay <= pool. The
        # single-request path is unchanged (padding_batch_size=1 -> graph[1]);
        # only the 2-on-one-rank case now finds a graph. HBM cost is the extra
        # batch-2 graph workspace + state pool, acceptable on the A3 box.
        mb = self.cache_config.max_batches
        return _get_capture_batch_size_impl(mb)


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    is_mla: bool


_graph_params: Optional[GraphParams] = None
_graph_capture_sizes: set[int] = None


def set_graph_params(aclgraph_capture_sizes: set[int]):
    global _graph_params
    global _graph_capture_sizes
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        events={size: [] for size in aclgraph_capture_sizes},
        workspaces={size: None for size in aclgraph_capture_sizes},
        handles={size: [] for size in aclgraph_capture_sizes},
        attn_params={size: [] for size in aclgraph_capture_sizes},
        is_mla=False,
    )
    _graph_capture_sizes = aclgraph_capture_sizes


def get_graph_params():
    return _graph_params


def clear_graph_params():
    """Clear global graph params and release references to KV cache tensors."""
    global _graph_params
    global _graph_capture_sizes
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
        _graph_capture_sizes = None
        # 清除 lru_cache，使下次推理时 _get_capture_batch_size_impl
        # 重新执行并调用 set_graph_params 干净重建
        _get_capture_batch_size_impl.cache_clear()


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
