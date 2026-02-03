# Copyright (c) 2024, DeepLink. All rights reserved.
import functools
import math
import os
import torch
import weakref
from typing import Callable, List

from lmdeploy.pytorch.backends.dlinfer.moe import DlinferFusedMoEImpl
from lmdeploy.pytorch.models.chatglm2 import SelfAttention
from lmdeploy.pytorch.engine import logits_process

from dlinfer.vendor.ascend.utils import SocVersion

# moe
from lmdeploy.pytorch.nn.moe import base
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
import lmdeploy.pytorch.distributed as dist

from lmdeploy.pytorch.disagg.messages import (
    AssignmentInstruct,
    DistServeRegisterMRMessage,
    MigrationAssignment,
    MigrationExecutionBatch,
)
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

import torch_npu
from torch_npu.profiler import profile as npu_profile


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

from lmdeploy.pytorch.engine.model_agent.profiler import AgentProfiler


def _build_npu_profiler(self):
    from lmdeploy.pytorch import envs

    activities = []
    if envs.torch_profile_cpu:
        activities.append(torch_npu.profiler.ProfilerActivity.CPU)
    if envs.torch_profile_cuda:
        activities.append(torch_npu.profiler.ProfilerActivity.NPU)
    if len(activities) > 0:
        logger.warning(
            f"Profiler start on {self.name}. "
            "Please Note that profiling might harm performance."
        )
        profiler = npu_profile(activities=activities)
        return profiler
    else:
        return None


AgentProfiler._build_profiler = _build_npu_profiler

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
