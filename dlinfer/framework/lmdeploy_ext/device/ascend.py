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


SelfAttention._fill_rope = ascend_chatglm2_fill_rope

########## below is for ascend310P ##########

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
        _try_to_cuda,
        msg_with_rank,
        AutoModelAgent,
        BaseModelAgent,
    )
    from lmdeploy.pytorch.engine.cache_engine import CacheEngine
    from lmdeploy.pytorch.models.patch import (
        update_custom_module_map,
        build_patched_model,
        add_adapters,
    )
    from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader

    logger = get_logger("lmdeploy")

    async def _async_step_background_310P(
        self,
        inputs: ModelInputs,
        swap_in_map: Dict,
        swap_out_map: Dict,
        loop_count: int,
        all_ids: torch.Tensor = None,
        guided_input_ids: torch.Tensor = None,
        sampling_inputs: SamplingInputs = None,
        num_appendable_ids: torch.LongTensor = None,
        num_ignore_eos: torch.LongTensor = None,
        return_logits: bool = False,
        is_dummy: bool = False,
        sync_long_context: bool = False,
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
                    [
                        guided_input_ids,
                        next_token_ids[:, None].to(guided_input_ids.device),
                    ],
                    1,
                )
            if sampling_inputs.random_offsets is not None:
                sampling_inputs.random_offsets += 1

        async def __await_distworker(worker, timeout: float = 0.001):
            while not worker.is_completed():
                await asyncio.sleep(timeout)
            worker.wait()

        # dist tools
        dist_ctx = get_dist_manager().current_context()
        rank = dist_ctx.rank
        tp = dist_ctx.tp
        dp = dist_ctx.dp

        logger.info(
            f"<ForwardTask> rank[{rank}]: "
            f"batch_size={inputs.seq_length.size(0)} "
            f"num_tokens={inputs.input_ids.size(-1)}"
        )

        is_decoding = inputs.is_decoding
        eager_mode = self.backend_config.eager_mode
        if dp > 1:
            if is_decoding and not eager_mode:
                batch_size = inputs.seq_length.numel()
                all_batch_sizes = torch.tensor([0] * dp, device="cuda")
                lc_handle = dist.all_gather_into_tensor(
                    all_batch_sizes,
                    all_batch_sizes.new_tensor(batch_size),
                    async_op=True,
                )
            else:
                all_sync_flags = torch.tensor([False] * dp, device="cuda")
                lc_handle = dist.all_gather_into_tensor(
                    all_sync_flags,
                    torch.tensor(sync_long_context, device="cuda"),
                    async_op=True,
                )

        non_blocking = True
        inputs = _try_to_cuda(inputs, non_blocking=non_blocking)
        all_ids = _try_to_cuda(all_ids, non_blocking=non_blocking)
        guided_input_ids = _try_to_cuda(guided_input_ids, non_blocking=non_blocking)
        sampling_inputs = _try_to_cuda(sampling_inputs, non_blocking=non_blocking)
        num_appendable_ids = _try_to_cuda(num_appendable_ids, non_blocking=non_blocking)
        num_ignore_eos = _try_to_cuda(num_ignore_eos, non_blocking=non_blocking)

        self.stream.synchronize()

        if dp > 1:
            if is_decoding and not eager_mode:
                await __await_distworker(lc_handle)
                padding_batch_size = all_batch_sizes.cpu().max().item()
                meta = self.patched_model.get_meta()
                meta.padding_batch_size = padding_batch_size
                logger.debug(f"padding_batch_size={padding_batch_size}")
            else:
                await __await_distworker(lc_handle)
                sync_long_context = all_sync_flags.any()
                logger.debug(f"sync_long_context={sync_long_context}")
            inputs.build_dp_meta()
            inputs = self.patched_model.update_inputs(inputs)
        else:
            sync_long_context = False

        need_output = dp > 1 or rank % tp == 0

        for idx in range(loop_count):
            # inference
            logger.debug(f"<ForwardTask> rank[{rank}]: model forward [{idx}].")
            output = await self._async_model_forward(
                inputs,
                swap_in_map=swap_in_map,
                swap_out_map=swap_out_map,
                return_logits=return_logits,
                sync_long_context=sync_long_context,
            )
            logits = output["logits"]
            logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

            if is_dummy:
                self._out_que.put_nowait(None)
                continue

            need_broadcast_next = dp == 1 and tp > 1 and idx < loop_count - 1
            if need_output:
                # sampling
                logger.debug(f"<ForwardTask> rank[{rank}]: Sampling [{idx}].")
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
                # Avoid adding the ADInplaceOrView dispatch key to `next_token_ids`,
                # as it can trigger recompilation on different ranks when using torch.compile.
                with torch.inference_mode():
                    next_token_ids = torch.empty_like(num_ignore_eos)
                stopped = None

            if need_broadcast_next:
                logger.debug(
                    f"<ForwardTask> rank[{rank}]: synchornize token ids [{idx}]"
                )
                # NOTE: Ascend310P does not support broadcast, so we use need to use gloo for broadcast next_token_ids and then transfer it to npu
                tp_cpu_group = dist_ctx.tp_cpu_group
                original_device = next_token_ids.device
                next_token_ids = next_token_ids.cpu()
                dist.broadcast(next_token_ids, src=rank // tp * tp, group=tp_cpu_group)
                next_token_ids = next_token_ids.to(original_device)

            # send output
            model_metas = output.get("model_metas")
            if need_output:
                event = torch.cuda.Event()
                event.record()
                output = dict(
                    next_token_ids=next_token_ids,
                    logits=logits if return_logits else None,
                    stopped=stopped,
                    model_metas=model_metas,
                    event=event,
                )
                logger.debug(f"<ForwardTask> rank[{rank}]: Output [{idx}]")
                self._out_que.put_nowait(output)

            # update for next loop
            if is_decoding and idx < loop_count - 1:
                swap_in_map = dict()
                swap_out_map = dict()
                inputs.model_metas = model_metas
                __update_inputs(next_token_ids)

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
        build patched model.
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
        patched_model = build_patched_model(self.model_config, device=device)
        logger.debug(msg_with_rank(rank, "loading weights."))
        load_model_weights_310P(patched_model, model_path, device=device)
        if adapters is not None:
            logger.debug(msg_with_rank(rank, "loading adapters."))
            add_adapters(
                patched_model, adapters, dtype=self.model_config.dtype, device=device
            )
        self.patched_model = patched_model

    # Ascend310P dose't support broadcast for now, so we need to use gloo for broadcast next_token_ids and then transfer it to npu
    AutoModelAgent._async_step_background = _async_step_background_310P
    # Ascend310P requires kv_cache to be acl NZ format. So allocate gpu cache in NZ format.
    CacheEngine._allocate_cache = _allocate_cache_310P
    # We convert Linear weight to NZ format on Ascend310P device defaultly in graph mode.
    # However, vision_model part is not compiled in graph mode, so we skip converting weights of vision_model part.
    BaseModelAgent._build_model = _build_model_310P
