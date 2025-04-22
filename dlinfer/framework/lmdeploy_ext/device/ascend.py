# Copyright (c) 2024, DeepLink. All rights reserved.
import torch

from lmdeploy.pytorch.models.chatglm2 import SelfAttention

from dlinfer.vendor.ascend.utils import SocVersion


@staticmethod
def ascend_chatglm2_fill_rope(states: torch.Tensor, rope: torch.Tensor):
    """fill rope."""
    rope_part = states.chunk(2, -1)[1]
    rope = rope.unflatten(-1, (2, -1))
    rope = rope.transpose(-2, -1).flatten(-2, -1)
    states = torch.cat([rope_part, rope], dim=-1)

    return states


if SocVersion.is_Ascend310P():
    # Layz import for Ascend310P
    import asyncio
    from typing import Dict
    import torch.distributed as dist
    from lmdeploy.utils import get_logger
    from lmdeploy.pytorch.model_inputs import ModelInputs
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.engine.model_agent import (
        _batch_stopping_criteria,
        AutoModelAgent,
    )
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.engine.cache_engine import CacheEngine

    logger = get_logger("lmdeploy")


async def _async_step_background_310P(
    self,
    inputs: ModelInputs,
    swap_in_map: Dict,
    swap_out_map: Dict,
    all_ids: torch.Tensor,
    guided_input_ids: torch.Tensor,
    sampling_inputs: SamplingInputs,
    num_appendable_ids: torch.LongTensor,
    num_ignore_eos: torch.LongTensor,
    loop_count: int,
    return_logits: bool,
    output_que: asyncio.Queue,
):
    """
    asyc forward task.
    # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
    # This mock for properly broadcasting next_token_ids on Ascend 310P device.
    """

    def __update_inputs(next_token_ids):
        """update inputs."""
        nonlocal all_ids, guided_input_ids
        inputs.update(next_token_ids)
        if all_ids is not None:
            all_ids = torch.cat(
                [all_ids, next_token_ids[:, None].to(all_ids.device)], 1
            )
        if guided_input_ids is not None:
            guided_input_ids = torch.cat(
                [guided_input_ids, next_token_ids[:, None].to(guided_input_ids.device)],
                1,
            )
        if sampling_inputs.random_offsets is not None:
            sampling_inputs.random_offsets += 1

    logger.debug(
        "<ForwardTask>: "
        f"batch_size={inputs.seq_length.size(0)} "
        f"num_tokens={inputs.input_ids.size(-1)}"
    )
    non_blocking = True
    inputs = inputs.to_device("cuda", non_blocking=non_blocking)
    is_decoding = inputs.is_decoding
    if all_ids is not None:
        all_ids = all_ids.cuda(non_blocking=non_blocking)
    if guided_input_ids is not None:
        guided_input_ids = guided_input_ids.cuda(non_blocking=non_blocking)
    sampling_inputs = sampling_inputs.to_device("cuda", non_blocking=non_blocking)
    num_appendable_ids = num_appendable_ids.cuda(non_blocking=non_blocking)
    num_ignore_eos = num_ignore_eos.cuda(non_blocking=non_blocking)

    self.stream.synchronize()

    # dist tools
    dist_ctx = get_dist_manager().current_context()
    rank = dist_ctx.rank
    tp = dist_ctx.tp

    for idx in range(loop_count):
        # inference
        output = await self._async_model_forward(
            inputs,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            return_logits=return_logits,
        )
        logits = output["logits"]
        logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

        if rank % tp == 0:
            # sampling
            next_token_ids = await self.async_sampling_logits(
                logits,
                all_ids,
                guided_input_ids,
                sampling_inputs,
                inputs,
                num_ignore_eos > 0,
            )
            num_ignore_eos = num_ignore_eos - 1

            # stopping criteria
            stopped, num_appendable_ids = _batch_stopping_criteria(
                next_token_ids, sampling_inputs.stop_words, num_appendable_ids
            )
        else:
            next_token_ids = torch.empty_like(num_ignore_eos)
            stopped = None

        if tp > 1 and idx < loop_count - 1:
            # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
            tp_cpu_group = dist_ctx.tp_cpu_group
            original_device = next_token_ids.device
            next_token_ids = next_token_ids.cpu()
            dist.broadcast(next_token_ids, src=rank // tp * tp, group=tp_cpu_group)
            next_token_ids = next_token_ids.to(original_device)

        # send output
        model_metas = output.get("model_metas")
        if rank % tp == 0:
            event = torch.cuda.Event()
            event.record()
            output = dict(
                next_token_ids=next_token_ids,
                logits=logits if return_logits else None,
                stopped=stopped,
                model_metas=model_metas,
                event=event,
            )
            output_que.put_nowait(output)

        # update for next loop
        if is_decoding and idx < loop_count - 1:
            swap_in_map = dict()
            swap_out_map = dict()
            inputs.model_metas = model_metas
            __update_inputs(next_token_ids)


@classmethod
def build_310P(cls, rank: int = 0, tp: int = 1, dp: int = 1, ccl_backend: str = "nccl"):
    """
    build dist context.
    # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
    # We need to inistialize the dist group for gloo backend as tp_cpu_group for next_token_ids broadcast.
    """
    from datetime import timedelta

    timeout = timedelta(days=35600)

    world_size = cls.get_world_size(tp, dp)
    if world_size == 1:
        return DistContext()

    assert dist.is_initialized()
    # world(assume world group is gloo)
    world_cpu_group = dist.GroupMember.WORLD

    # tp
    tp_gpu_group = None
    tp_rank = rank % tp
    if tp > 1:
        tp_rank0 = rank // tp
        tp_ranks = list(range(tp_rank0, tp_rank0 + tp))
        tp_gpu_group = dist.new_group(
            ranks=tp_ranks, timeout=timeout, backend=ccl_backend
        )
        tp_cpu_group = dist.new_group(ranks=tp_ranks, timeout=timeout, backend="gloo")

    # dp
    dp_gpu_group = None
    if dp > 1 and rank % tp == 0:
        dp_ranks = list(range(0, world_size, tp))
        dp_gpu_group = dist.new_group(
            ranks=dp_ranks, timeout=timeout, backend=ccl_backend
        )

    context = DistContext(
        rank=rank,
        world_size=world_size,
        tp=tp,
        dp=dp,
        tp_rank=tp_rank,
        world_cpu_group=world_cpu_group,
        tp_cpu_group=tp_cpu_group,
        tp_gpu_group=tp_gpu_group,
        dp_cpu_group=None,
        dp_gpu_group=dp_gpu_group,
    )
    return context


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


SelfAttention._fill_rope = ascend_chatglm2_fill_rope
if SocVersion.is_Ascend310P():
    # Ascend310P dose't support broadcast for now, so we need to use gloo for broadcast next_token_ids and then transfer it to npu
    DistContext.build = build_310P
    AutoModelAgent._async_step_background = _async_step_background_310P
    # Ascend310P requires kv_cache to be acl NZ format. So allocate gpu cache in NZ format.
    CacheEngine._allocate_cache = _allocate_cache_310P
