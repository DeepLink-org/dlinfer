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
    BaseModelAgent._broadcast_next_token = _broadcast_next_token_310P
    # Ascend310P requires kv_cache to be acl NZ format. So allocate gpu cache in NZ format.
    CacheEngine._allocate_cache = _allocate_cache_310P
    # We convert Linear weight to NZ format on Ascend310P device defaultly in graph mode.
    # However, vision_model part is not compiled in graph mode, so we skip converting weights of vision_model part.
    BaseModelAgent._build_model = _build_model_310P
