# Copyright (c) 2024, DeepLink. All rights reserved.
import functools
import math
import os
import torch
import weakref
from typing import Callable, Dict, List, Optional, Any, Literal, Tuple

from lmdeploy.pytorch.backends.dlinfer.moe import DlinferFusedMoEImpl
from lmdeploy.pytorch.models.chatglm2 import SelfAttention
from lmdeploy.pytorch.engine import logits_process

from dlinfer.vendor.ascend.utils import SocVersion

# moe
from lmdeploy.pytorch.nn.moe import base
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
import lmdeploy.pytorch.distributed as dist


# cache engine
from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.engine import cache_engine
from lmdeploy.pytorch.config import ModelConfig, CacheConfig
import lmdeploy.pytorch.engine.executor.base as executor_base  # noqa: E402
import lmdeploy.pytorch.engine.model_agent as model_agent  # noqa: E402
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
)
from lmdeploy.pytorch.disagg.messages import (
    AssignmentInstruct,
    DistServeRegisterMRMessage,
    MigrationAssignment,
    MigrationExecutionBatch,
)
from lmdeploy.utils import get_logger


def rl_update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
    """Update weights."""
    return gate_up_weights, down_weights


if os.getenv("DLINFER_RESET_MOE_UPDATE_WEIGHTS", "0") == "1":
    DlinferFusedMoEImpl.update_weights = rl_update_weights


@staticmethod
def ascend_chatglm2_fill_rope(states: torch.Tensor, rope: torch.Tensor):
    """fill rope."""
    rope_part = states.chunk(2, -1)[1]
    rope = rope.unflatten(-1, (2, -1))
    rope = rope.transpose(-2, -1).flatten(-2, -1)
    states = torch.cat([rope_part, rope], dim=-1)

    return states


SelfAttention._fill_rope = ascend_chatglm2_fill_rope


# modify bad words process for aclgraph performance
def _process_bad_words_(
    scores: torch.Tensor,
    bad_words: torch.LongTensor,
    mask: torch.BoolTensor,
    filter_value: float = -99999.9999,
):
    """Process bad words."""
    filtered_scores = scores.gather(1, bad_words)
    filtered_scores = mask.to(filtered_scores.dtype) * filter_value + filtered_scores
    scores.scatter_(1, bad_words, filtered_scores)
    return scores


logits_process._process_bad_words_ = _process_bad_words_


# patch MoEForwardDPTP
hidden_states_gather_buffer = None
topk_weights_gather_buffer = None
topk_ids_gather_buffer = None


@functools.lru_cache
def _eager_mode():
    from lmdeploy.pytorch.backends.dlinfer.ascend import AscendOpsBackend

    return not AscendOpsBackend.enable_graph


@functools.lru_cache
def _get_max_batch_size():
    from lmdeploy.pytorch.backends.dlinfer.ascend import AscendOpsBackend

    return AscendOpsBackend.max_batches


def _clear_moe_comm_buffers():
    """Clear module-level MoE comm buffers."""
    global hidden_states_gather_buffer
    global topk_weights_gather_buffer
    global topk_ids_gather_buffer
    hidden_states_gather_buffer = None
    topk_weights_gather_buffer = None
    topk_ids_gather_buffer = None


class AscendMoEForwardDPTP:

    def __init__(self, gemm_func: Callable, max_tokens_per_round: int = 3000):
        """MoE forward dp tp."""
        self.gemm_func = gemm_func
        self.dist_ctx = get_dist_manager().current_context()
        self.dist_config = self.dist_ctx.dist_config
        self.tp = self.dist_config.moe_tp
        self.attn_tp = self.dist_config.attn_tp

        tp_group = self.dist_ctx.moe_tp_group
        self.rank = tp_group.rank
        self.gather_rank = self.rank // self.attn_tp
        self.gather_group = tp_group.gpu_gather_group
        self.tp_group = tp_group.gpu_group

        # self.max_tokens_per_round = max_tokens_per_round * self.attn_tp // self.tp // 2
        self.max_tokens_per_round = max_tokens_per_round
        self.use_comm_buffer = None

        # When instance is GC'ed, clear module-level global buffers
        self._finalizer = weakref.finalize(self, _clear_moe_comm_buffers)

    def _init_comm_buffer(
        self,
        step_ctx,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        self.max_batch_size = _get_max_batch_size()
        self.hidden_size = hidden_states.size(-1)
        self.topk = topk_ids.size(-1)
        self.dp_size = len(step_ctx.dp_meta.moe_tp_sizes)

        global hidden_states_gather_buffer
        global topk_weights_gather_buffer
        global topk_ids_gather_buffer

        hidden_states_gather_buffer = hidden_states.new_empty(
            self.max_batch_size * self.dp_size * self.hidden_size
        )
        topk_weights_gather_buffer = topk_weights.new_empty(
            self.max_batch_size * self.dp_size * self.topk
        )
        topk_ids_gather_buffer = topk_ids.new_empty(
            self.max_batch_size * self.dp_size * self.topk
        )

    def all_gather(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        tp_sizes: List[int],
    ):
        """All gather."""
        hidden_states = dist.gather_by_tp_sizes(
            hidden_states, tp_sizes, group=self.gather_group, async_op=False
        )
        topk_weights = dist.gather_by_tp_sizes(
            topk_weights, tp_sizes, group=self.gather_group, async_op=False
        )
        topk_ids = dist.gather_by_tp_sizes(
            topk_ids, tp_sizes, group=self.gather_group, async_op=False
        )
        return hidden_states, topk_weights, topk_ids

    def reduce_scatter(
        self, hidden_states: torch.Tensor, out_states: torch.Tensor, tp_sizes: List[int]
    ):
        """Reduce scatter."""
        hidden_states_list = list(hidden_states.split(tp_sizes, -2))
        cur_out_states = hidden_states_list[self.gather_rank]
        out_states.copy_(cur_out_states)
        hidden_states_list = [
            item for item in hidden_states_list for _ in range(self.attn_tp)
        ]
        hidden_states_list[self.rank] = out_states
        dist.reduce_scatter(
            out_states, hidden_states_list, group=self.tp_group, async_op=False
        )
        return out_states

    def _gemm_and_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output_states: torch.Tensor,
        tp_sizes: List[int],
    ):
        """Gemm and reduce scatter."""
        cur_out = self.gemm_func(hidden_states, topk_weights, topk_ids)
        return self.reduce_scatter(cur_out, output_states, tp_sizes)

    def forward_decode(
        self,
        step_ctx,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """forward."""
        tp_sizes = step_ctx.dp_meta.moe_tp_sizes

        if self.use_comm_buffer:
            cur_hidden_states = hidden_states_gather_buffer[
                : hidden_states.size(0) * self.dp_size * self.hidden_size
            ].view(hidden_states.size(0) * self.dp_size, self.hidden_size)
            cur_topk_weights = topk_weights_gather_buffer[
                : topk_weights.size(0) * self.dp_size * self.topk
            ].view(topk_weights.size(0) * self.dp_size, self.topk)
            cur_topk_ids = topk_ids_gather_buffer[
                : topk_ids.size(0) * self.dp_size * self.topk
            ].view(topk_ids.size(0) * self.dp_size, self.topk)

            torch.distributed.all_gather_into_tensor(
                cur_hidden_states,
                hidden_states,
                group=self.gather_group,
                async_op=False,
            )
            torch.distributed.all_gather_into_tensor(
                cur_topk_weights, topk_weights, group=self.gather_group, async_op=False
            )
            torch.distributed.all_gather_into_tensor(
                cur_topk_ids, topk_ids, group=self.gather_group, async_op=False
            )
        else:
            cur_hidden_states, cur_topk_weights, cur_topk_ids = self.all_gather(
                hidden_states, topk_weights, topk_ids, tp_sizes
            )

        # MoE gemm
        cur_out = self.gemm_func(cur_hidden_states, cur_topk_weights, cur_topk_ids)
        output_states = dist.reduce_scatter_by_tp_sizes(
            cur_out, self.rank, tp_sizes, group=self.tp_group
        )
        return output_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """forward."""
        step_ctx = get_step_ctx_manager().current_context()
        if step_ctx.is_decoding:
            return self.forward_decode(step_ctx, hidden_states, topk_weights, topk_ids)

        # lazy init comm buffer
        if self.use_comm_buffer is None:
            self.eager_mode = _eager_mode()
            self.use_comm_buffer = not self.eager_mode

            if self.use_comm_buffer:
                self._init_comm_buffer(step_ctx, hidden_states, topk_weights, topk_ids)

        def __slice_tensor(tensor: torch.Tensor, slice_size: int):
            """Slice tensor."""
            cur_tensor = tensor[:slice_size]
            tensor = tensor[slice_size:]
            return cur_tensor, tensor

        def __slice_and_gather():
            """Slice and gather."""
            nonlocal hidden_states, topk_weights, topk_ids, tp_sizes, output_states
            cur_tp_sizes = tp_sizes.minimum(max_tokens_per_round)
            tp_sizes -= cur_tp_sizes
            cur_tp_sizes = cur_tp_sizes.tolist()

            slice_size = cur_tp_sizes[self.gather_rank]
            cur_hidden_states, hidden_states = __slice_tensor(hidden_states, slice_size)
            cur_topk_weights, topk_weights = __slice_tensor(topk_weights, slice_size)
            cur_topk_ids, topk_ids = __slice_tensor(topk_ids, slice_size)
            cur_output, output_states = __slice_tensor(output_states, slice_size)

            total_tokens = sum(cur_tp_sizes)
            sh = (
                *cur_hidden_states.shape[:-2],
                total_tokens,
                *cur_hidden_states.shape[-1:],
            )
            sw = (
                *cur_topk_weights.shape[:-2],
                total_tokens,
                *cur_topk_weights.shape[-1:],
            )
            si = (*cur_topk_ids.shape[:-2], total_tokens, *cur_topk_ids.shape[-1:])

            b_hidden = cur_hidden_states.new_empty(sh)
            b_weights = cur_topk_weights.new_empty(sw)
            b_ids = cur_topk_ids.new_empty(si)

            # map gather-group local index -> global rank for broadcast src
            from lmdeploy.pytorch.distributed import get_world_rank

            world_size, global_rank = get_world_rank()
            tp = self.tp
            attn_tp = self.attn_tp
            tp_group_id = global_rank // tp
            group_start = tp_group_id * tp
            gather_group_id = (global_rank - group_start) % attn_tp
            g_start = group_start + gather_group_id
            g_ranks = list(range(world_size))[g_start : (g_start + tp) : attn_tp]

            offset = 0
            for src_idx, sz in enumerate(cur_tp_sizes):
                h_slice = b_hidden.narrow(-2, offset, sz)
                w_slice = b_weights.narrow(-2, offset, sz)
                i_slice = b_ids.narrow(-2, offset, sz)

                if src_idx == self.gather_rank:
                    h_slice.copy_(cur_hidden_states)
                    w_slice.copy_(cur_topk_weights)
                    i_slice.copy_(cur_topk_ids)

                src_global = g_ranks[src_idx]

                dist.broadcast(
                    h_slice, src_global, group=self.gather_group, async_op=False
                )
                dist.broadcast(
                    w_slice, src_global, group=self.gather_group, async_op=False
                )
                dist.broadcast(
                    i_slice, src_global, group=self.gather_group, async_op=False
                )

                offset += sz
            cur_hidden_states, cur_topk_weights, cur_topk_ids = (
                b_hidden,
                b_weights,
                b_ids,
            )

            return dict(
                hidden_states=cur_hidden_states,
                topk_weights=cur_topk_weights,
                topk_ids=cur_topk_ids,
                output_states=cur_output,
                tp_sizes=cur_tp_sizes,
            )

        tp_sizes = step_ctx.dp_meta.moe_tp_sizes
        tp_sizes = torch.tensor(tp_sizes)
        max_tokens_per_round = tp_sizes.new_tensor(self.max_tokens_per_round)

        output_states = torch.empty_like(hidden_states)
        return_states = output_states

        # pre
        cur_inputs = __slice_and_gather()

        # main loop
        while tp_sizes.sum() > 0:
            next_inputs = __slice_and_gather()
            self._gemm_and_reduce_scatter(**cur_inputs)
            cur_inputs = next_inputs

        # post
        self._gemm_and_reduce_scatter(**cur_inputs)
        return return_states


base.MoEForwardDPTP = AscendMoEForwardDPTP

########## below is for ascend310P ##########

if SocVersion.is_Ascend310P():
    # Layz import for Ascend310P
    import torch.distributed as dist
    from lmdeploy.utils import get_logger
    from lmdeploy.pytorch.distributed import get_dist_manager, DistContext
    from lmdeploy.pytorch.engine.model_agent import (
        msg_with_rank,
        BaseModelAgent,
    )
    from lmdeploy.pytorch.engine.cache_engine import CacheEngine
    from lmdeploy.pytorch.models.patch import (
        update_custom_module_map,
        build_patched_model,
        add_adapters,
    )
    from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader
    from lmdeploy.pytorch.disagg.config import EngineRole

    logger = get_logger("lmdeploy")

    def _broadcast_next_token_310P(
        self, next_token_ids: torch.Tensor, dist_ctx: DistContext = None
    ):
        # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
        # This mock for properly broadcasting next_token_ids on Ascend 310P device.
        if dist_ctx is None:
            dist_ctx = get_dist_manager().current_context()
        if self.cache_config.role == EngineRole.Decode:
            next_token_ids = next_token_ids.cpu()
            tp_cpu_group = dist_ctx.tp_cpu_group
            dist.all_reduce(next_token_ids, op=dist.ReduceOp.SUM, group=tp_cpu_group)
        else:
            # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
            tp_cpu_group = dist_ctx.tp_cpu_group
            original_device = next_token_ids.device
            next_token_ids = next_token_ids.cpu()
            dist.broadcast(next_token_ids, src=0, group=tp_cpu_group)
            next_token_ids = next_token_ids.to(original_device)
        return next_token_ids

    def _allocate_cache_310P(self, num_blocks: int, device: torch.device):
        """
        allocate cache implement.
        # NOTE. Ascend300I duo devices require kv_cache to be acl NZ format.
        """
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        num_layers = self.num_layers
        kv_cache_dtype = self.kv_cache_dtype

        if device != "cpu":
            import torch_npu

            key_cache = torch_npu.empty_with_format(
                size=(num_layers, num_blocks, *key_block_shape),
                dtype=kv_cache_dtype,
                device="npu",
                acl_format=29,  # 29 for acl NZ format
            )
            value_cache = torch_npu.empty_with_format(
                size=(num_layers, num_blocks, *value_block_shape),
                dtype=kv_cache_dtype,
                device="npu",
                acl_format=29,
            )
        else:
            key_cache = torch.empty(
                size=(num_layers, num_blocks, *key_block_shape),
                dtype=kv_cache_dtype,
                device=device,
            )
            value_cache = torch.empty(
                size=(num_layers, num_blocks, *value_block_shape),
                dtype=kv_cache_dtype,
                device=device,
            )

        output = (key_cache, value_cache)

        if self.cache_config.quant_policy in (4, 8):
            dtype = self.model_config.dtype
            key_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *key_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            val_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *value_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            output = output + (key_sz_cache, val_sz_cache)

        return output

    @torch.inference_mode()
    def load_model_weights_310P(
        model: torch.nn.Module,
        checkpoint_path: str,
        prefix: str = None,
        device: torch.device = None,
    ):
        """Loading model weights."""
        loader = ModelWeightLoader(checkpoint_path, prefix=prefix)
        loader.load_model_weights(model, device=device)
        model.eval()
        # NOTE: Ascend310P convert Linear weight to NZ format defaultly in graph mode.
        # However, vision_model part is not compiled in graph mode, so we skip converting weights of vision_model part.
        # This is a workaround for Ascend310P.
        for name, mod in model.named_modules():
            if (
                not hasattr(mod, "update_weights")
                or name.startswith("vision_model")
                or name.startswith("visual")
            ):
                continue
            mod.update_weights()

    def _build_model_310P(self):
        """
        Build patched model.
        NOTE: Ascend310P convert Linear weight to NZ format defaultly in graph mode.
        However, vision_model part is not compiled in graph mode, so we skip converting weights of vision_model part.
        """
        model_path = self.model_path
        adapters = self.adapters
        device = self.device
        rank = self.rank
        custom_module_map = self.model_config.custom_module_map
        if custom_module_map is not None:
            update_custom_module_map(custom_module_map)
        logger.debug(msg_with_rank(rank, "build model."))
        patched_model = build_patched_model(
            self.model_config, device=device, model_format=self.misc_config.model_format
        )
        logger.debug(msg_with_rank(rank, "loading weights."))
        if not self.misc_config.empty_init:
            load_model_weights_310P(patched_model, model_path, device=device)
        if adapters is not None:
            logger.debug(msg_with_rank(rank, "loading adapters."))
            add_adapters(
                patched_model, adapters, dtype=self.model_config.dtype, device=device
            )
        self.patched_model = patched_model

    # Ascend310P dose't support broadcast for now, so we need to use gloo for broadcast next_token_ids and then transfer it to npu
    BaseModelAgent._broadcast_next_token = _broadcast_next_token_310P
    # Ascend310P requires kv_cache to be acl NZ format. So allocate gpu cache in NZ format.
    CacheEngine._allocate_cache = _allocate_cache_310P
    # We convert Linear weight to NZ format on Ascend310P device defaultly in graph mode.
    # However, vision_model part is not compiled in graph mode, so we skip converting weights of vision_model part.
    BaseModelAgent._build_model = _build_model_310P


##### patch cache engine #####
logger = get_logger("lmdeploy")


class AscendCacheEngine:
    """Host and Device memory maintainer.

    Args:
        cache_config (CacheConfig): config of the cache information.
        model_config (ModelConfig): config of the model.
        rank (int): distribution rank, 0 on non-distributed environment.
        world_size (int): distribution world size, 1 on non-distributed
            environment.
        cache_stream (torch.cuda.Stream): the stream used for cache engine swap,
            if set to None, it's created in CacheEngine.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        rank: int = 0,
        tp_rank: int = 0,
        world_size: int = 1,
        cache_stream: torch.cuda.Stream = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.tp_rank = tp_rank
        self.cache_config = cache_config
        self.model_config = model_config

        self.block_size = cache_config.block_size
        self.num_layers = model_config.num_layers
        self.kv_cache_dtype = model_config.dtype
        if cache_config.quant_policy > 0:
            if self.cache_config.device_type in ["cuda"]:
                self.kv_cache_dtype = torch.uint8
            elif self.cache_config.device_type in ["ascend", "npu"]:
                self.kv_cache_dtype = torch.int8
            else:
                raise ValueError(
                    f"unsupported device_type {self.cache_config.device_type}"
                )

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()

        self.migration_backend_impl: Optional[MigrationBackendImpl] = None

        # Initialize the stream for caching operations.
        self.cache_stream = cache_stream or torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = torch.cuda.Event()

        logger.debug(
            f"Initialize cache engine with {cache_config.num_gpu_blocks}"
            f" gpu blocks and {cache_config.num_cpu_blocks} cpu blocks."
        )

    @property
    def cpu_cache(self):
        """Gpu cache."""
        return self.local_cpu_cache

    @property
    def gpu_cache(self):
        """Gpu cache."""
        return self.local_gpu_cache

    @property
    def num_gpu_blocks(self):
        """Num gpu blocks."""
        return self.cache_config.num_gpu_blocks

    @property
    def num_cpu_blocks(self):
        """Num gpu blocks."""
        return self.cache_config.num_cpu_blocks

    @classmethod
    def _get_key_block_shape_impl(
        cls,
        model_config: ModelConfig,
        block_size: int,
        head_size: int,
        world_size: int = 1,
        quant_policy: Literal[0, 4, 8] = 0,
        local: bool = True,
    ):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local:
            assert (
                num_heads % world_size == 0
            ), f"num_heads: {num_heads}, world_size: {world_size}"
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert (
                head_size % 2 == 0
            ), f"head_size: {head_size}, quant_policy: {quant_policy}"
            head_size = head_size // 2
        return attn_backend.get_k_block_shape(block_size, num_heads, head_size, dtype)

    @classmethod
    def _get_value_block_shape_impl(
        cls,
        model_config: ModelConfig,
        block_size: int,
        head_size: int,
        world_size: int = 1,
        quant_policy: Literal[0, 4, 8] = 0,
        local: bool = True,
    ):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local:
            assert (
                num_heads % world_size == 0
            ), f"num_heads: {num_heads}, world_size: {world_size}"
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert (
                head_size % 2 == 0
            ), f"head_size: {head_size}, quant_policy: {quant_policy}"
            head_size = head_size // 2

        return attn_backend.get_v_block_shape(block_size, num_heads, head_size, dtype)

    def get_key_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """Get shape of key block."""
        head_size = self.model_config.k_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_key_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def get_value_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """Get shape of value block."""
        head_size = self.model_config.v_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_value_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def _allocate_cache(self, num_blocks: int, device: torch.device):
        """Allocate cache implement."""
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        num_layers = self.num_layers
        kv_cache_dtype = self.kv_cache_dtype

        key_cache = torch.empty(
            size=(num_layers, num_blocks, *key_block_shape),
            dtype=kv_cache_dtype,
            device=device,
        )
        value_cache = torch.empty(
            size=(num_layers, num_blocks, *value_block_shape),
            dtype=kv_cache_dtype,
            device=device,
        )

        output = (key_cache, value_cache)

        if self.cache_config.quant_policy in (4, 8):
            dtype = self.model_config.dtype
            key_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *key_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            val_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *value_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            output = output + (key_sz_cache, val_sz_cache)

        return output

    def allocate_gpu_cache(self):
        """Allocate caches on GPU."""
        caches = self._allocate_cache(self.num_gpu_blocks, "cuda")
        self.full_gpu_cache = caches
        self.local_gpu_cache = list(zip(*caches))
        return self.local_gpu_cache

    def allocate_cpu_cache(self):
        """Allocate caches on Host."""
        caches = self._allocate_cache(self.num_cpu_blocks, "cpu")

        self.full_cpu_cache = caches
        self.local_cpu_cache = list(zip(*caches))
        return self.local_cpu_cache

    @torch.inference_mode()
    def _swap(
        self,
        src: List[torch.Tensor],
        dst: List[torch.Tensor],
        src_to_dst: Dict[int, int],
    ):
        """Move caches from src memory to dst memory.

        Args:
            src (List[KVCache]): Source cache.
            dst (List[KVCache]): Destination cache.
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        BLOCKS_PER_COPY = 2
        num_copy = len(src_to_dst)
        src_idx, dst_idx = list(zip(*src_to_dst.items()))
        src_idx = torch.tensor(src_idx, device=src[0].device)
        dst_idx = torch.tensor(dst_idx, device=dst[0].device)
        with torch.cuda.stream(self.cache_stream):
            for scache, dcache in zip(src, dst):
                for idx in range(0, num_copy, BLOCKS_PER_COPY):
                    sidx = src_idx[idx : idx + BLOCKS_PER_COPY]
                    didx = dst_idx[idx : idx + BLOCKS_PER_COPY]
                    sdata = scache[:, sidx]
                    dcache.index_copy_(1, didx, sdata.to(dcache.device))
            self.events.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Host to Device.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.full_cpu_cache, self.full_gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Device to Host.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.full_gpu_cache, self.full_cpu_cache, src_to_dst)

    @classmethod
    def get_cache_block_size(
        cls,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        world_size: int = 1,
    ) -> int:
        """Get the required cache size of the model.

        Args:
            cache_config (CacheConfig): The config of the cache.
            model_config (ModelConfig): The config of the model.
            world_size (int): The world size for tensor parallelism.

        Return:
            int: Required memory size in bytes.
        """

        def _dtype_size(dtype: torch.dtype) -> int:
            """Get element size without using meta device (torch_npu compatible)."""
            return torch.tensor([], dtype=dtype).element_size()

        block_size = cache_config.block_size
        quant_policy = cache_config.quant_policy
        num_layers = model_config.num_layers

        key_head_size = model_config.k_head_dim
        value_head_size = model_config.v_head_dim
        if key_head_size is None:
            key_head_size = model_config.head_dim
        if value_head_size is None:
            value_head_size = model_config.head_dim

        key_shape = cls._get_key_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=key_head_size,
            world_size=world_size,
            local=True,
            quant_policy=quant_policy,
        )
        value_shape = cls._get_value_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=value_head_size,
            world_size=world_size,
            quant_policy=quant_policy,
            local=True,
        )

        if quant_policy == 0:
            dtype_size = _dtype_size(model_config.dtype)
            mem_key_block = math.prod(key_shape) * dtype_size
            mem_value_block = math.prod(value_shape) * dtype_size
        elif quant_policy in (4, 8):
            # KV cache uses uint8/int8, scale/zero uses model dtype
            kv_dtype = (
                torch.uint8 if cache_config.device_type in ["cuda"] else torch.int8
            )
            kv_dtype_size = _dtype_size(kv_dtype)
            scale_dtype_size = _dtype_size(model_config.dtype)

            key_scale_zero_shape = (*key_shape[:-1], 2)
            val_scale_zero_shape = (*value_shape[:-1], 2)

            mem_key_block = (
                math.prod(key_shape) * kv_dtype_size
                + math.prod(key_scale_zero_shape) * scale_dtype_size
            )
            mem_value_block = (
                math.prod(value_shape) * kv_dtype_size
                + math.prod(val_scale_zero_shape) * scale_dtype_size
            )
        else:
            raise ValueError(f"unsupported quant_policy {quant_policy}")

        total = num_layers * (mem_key_block + mem_value_block)
        return total

    """ Metheds for PD Disaggregation Begin. """

    def p2p_initialize(
        self, migration_init_request: DistServeInitRequest
    ) -> DistServeKVTransferEndpointInfo:
        if not self.migration_backend_impl:
            self.migration_backend_impl = MIGRATION_BACKENDS.module_dict[
                self.cache_config.migration_backend.name
            ]()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)
        for i, t in enumerate(self.full_gpu_cache):
            if t.numel() == 0:
                continue
            register_mr_request = DistServeRegisterMRMessage(
                protocol=migration_init_request.protocol,
                remote_engine_id=migration_init_request.remote_engine_id,
                mr_key=str(i),
                addr=t.data_ptr(),
                offset=t.storage_offset(),
                length=t.numel() * t.itemsize,
            )
            self.migration_backend_impl.register_memory_region(register_mr_request)
        return DistServeKVTransferEndpointInfo(
            protocol=migration_init_request.protocol,
            endpoint_info=json.dumps(
                self.migration_backend_impl.endpoint_info(
                    migration_init_request.remote_engine_id,
                    migration_init_request.protocol,
                )
            ),
        )

    def p2p_connect(
        self,
        remote_engine_id: str,
        migration_conn_request: List[DistServeKVTransferEndpointInfo],
    ):
        self.migration_backend_impl.p2p_connect(
            remote_engine_id, migration_conn_request[self.tp_rank]
        )

    async def migrate(self, migration_execution_inputs: MigrationExecutionBatch):

        def get_assignment_len():
            head_dim = self.model_config.get_head_size()
            num_heads = self.model_config.num_key_value_heads // self.world_size
            block_size = self.cache_config.block_size
            return head_dim * num_heads * block_size * self.model_config.dtype.itemsize

        assignment_len = get_assignment_len()
        layer_stride = self.cache_config.num_gpu_blocks * assignment_len

        def get_assignment_batch(
            mr_key, block_ids, assignment_len, layer_stride, remote_layer_stride
        ):
            return [
                AssignmentInstruct(
                    mr_key=mr_key,
                    target_offset=block_id[0] * assignment_len
                    + layer * remote_layer_stride,
                    source_offset=block_id[1] * assignment_len + layer * layer_stride,
                    length=assignment_len,
                )
                for layer in range(self.model_config.num_layers)
                for block_id in block_ids
            ]

        assignment_batch: List[Tuple[str, int, int, int]] = (
            []
        )  # mr_key, target, source, offset
        for migration_exe_req in migration_execution_inputs.requests:
            remote_engine_id = migration_exe_req[0]
            blocks_to_migration = migration_exe_req[1]
            remote_layer_stride = (
                self.migration_backend_impl.links[
                    remote_engine_id
                ].remote_engine_config.num_gpu_blocks
                * assignment_len
            )

            for i, t in enumerate(self.full_gpu_cache):
                if t.numel() == 0:
                    continue
                assignment_batch.extend(
                    get_assignment_batch(
                        str(i),
                        blocks_to_migration,
                        assignment_len,
                        layer_stride,
                        remote_layer_stride,
                    )
                )
        await self.migration_backend_impl.p2p_migrate(
            MigrationAssignment(
                protocol=migration_execution_inputs.protocol,
                remote_engine_id=remote_engine_id,
                batch=assignment_batch,
            )
        )

    """ Metheds for PD Disaggregation End. """


# Make sure all modules that already captured CacheEngine see the patched class.
cache_engine.CacheEngine = AscendCacheEngine
executor_base.CacheEngine = AscendCacheEngine
model_agent.CacheEngine = AscendCacheEngine


##### patch scheduler #####
##### workaround for uncompleted prefill_attention_with_kvcache #####
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.messages import MessageStatus
from lmdeploy.pytorch.engine.request import EventType


def _schedule_prefill_ascend(self, prealloc_size: int = 0):
    """Schedule prefill for Ascend devices.

    This patched version adds logic to handle prefill with kv-cache optimization.
    It ensures that prefill sequences are properly scheduled while respecting
    batch size and token count limits.
    """
    running = self.running
    swap_in_map = dict()
    swap_out_map = dict()
    copy_map = dict()
    token_count = sum([seq.num_token_ids for seq in running])
    max_batches = self.cache_config.max_batches

    def _reorder_waiting():
        """Reorder waiting sequences."""
        return self.seq_manager.get_sequences(MessageStatus.WAITING)

    def __evict_for_seq(seq, waiting):
        """Evict blocks for sequence."""
        while not self.block_manager.can_allocate(seq, prealloc_size):
            if not self._evict(running, waiting):
                return False
        return True

    def _to_running(seq):
        """Move sequence to running state."""
        running.append(seq)
        self.seq_manager.update_message_status(seq.msg_id, MessageStatus.RUNNING)
        nonlocal token_count
        token_count += seq.num_token_ids

    num_waiting = self.seq_manager.num_sequences(MessageStatus.WAITING)
    if len(running) >= max_batches or num_waiting == 0:
        return running, swap_in_map, swap_out_map, copy_map

    waiting = _reorder_waiting()
    prefill_with_kvcache = True
    while len(waiting) > 0 and len(running) < max_batches:
        seq = waiting.pop(0)
        if prefill_with_kvcache == False and seq.num_new_tokens > 0:
            break
        prefill_with_kvcache = False if seq.num_new_tokens == 0 else True

        if (
            len(running) > 0
            and token_count + seq.num_token_ids
            > self.cache_config.max_prefill_token_num
        ):
            break

        self.block_trie.match(seq)

        if not __evict_for_seq(seq, waiting):
            break

        # allocate session memory
        self.block_manager.allocate(seq, prealloc_size)
        self.block_trie.allocate(seq)
        if self.is_ssm:
            self.state_manager.allocate(seq)
        _to_running(seq)

        seq.record_event(EventType.SCHEDULED)
        if prefill_with_kvcache == True:
            break

    return running, swap_in_map, swap_out_map, copy_map


Scheduler._schedule_prefill = _schedule_prefill_ascend
