# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
import torch
from functools import lru_cache
from dlinfer.vendor import vendor_name

vendor = ["camb", "ascend"]


def fake_torch_compile(dynamic=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


def post_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    variance: torch.Tensor,
    eps: float,
    embed_dim: int,
    dtype: torch.dtype,
):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


def patch_compiled_func():
    import torch

    real_torch_compile = torch.compile
    torch.compile = fake_torch_compile
    from lmdeploy.pytorch.models import internvl, internvl3_hf

    internvl.pre_rms_norm = pre_rms_norm
    internvl.post_rms_norm = post_rms_norm
    internvl3_hf.pre_rms_norm = pre_rms_norm
    internvl3_hf.post_rms_norm = post_rms_norm
    torch.compile = real_torch_compile


def patch_async_sampling_logits():
    from torch.profiler import record_function
    from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
    from lmdeploy.pytorch.engine.logits_process import (
        SamplingInputs,
        FusedLogitsProcessor,
    )

    async def async_sampling_logits(
        self, logits: torch.Tensor, sampling_inputs: SamplingInputs
    ):
        """Sampling logits."""
        # record function does not support async function
        # so we can not decorate it on async_sampling_logits
        with record_function("sampling_logits"):
            logits = logits.to(torch.float32)
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=self.guided_decoding_manager,
            )
            origin_logits = logits
            logits, raw_logprobs = await logits_processor(origin_logits)
            next_token_ids = logits_processor.sampling(logits)
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )

        return next_token_ids, logprobs

    BaseModelAgent.async_sampling_logits = async_sampling_logits


##### patch dlinfer moe: DlinferSoftmaxTopKImpl #####
def patch_dlinfer_moe():
    import torch
    from lmdeploy.pytorch.kernels.dlinfer import moe_gating_topk_softmax
    from lmdeploy.pytorch.backends import moe as backends_moe
    from lmdeploy.pytorch.backends.dlinfer import moe as dlinfer_moe

    class PatchedDlinferSoftmaxTopKImpl(backends_moe.SoftmaxTopKImpl):
        """Dlinfer softmax topk implementation (patched with n_groups support)."""

        def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
            self.top_k = top_k
            self.dim = dim
            self.n_groups = n_groups

        def forward(self, x: torch.Tensor):
            if self.n_groups > 0:
                routing_weights = torch.softmax(x, dim=self.dim, dtype=torch.float32)
                assert (
                    routing_weights.shape[self.dim] % self.n_groups == 0
                ), f"{routing_weights.shape[self.dim]} cannot be divided by {self.n_groups}"
                per_group_top_k = self.top_k // self.n_groups
                group_size = routing_weights.shape[self.dim] // self.n_groups
                group_offsets = self.get_group_offsets(
                    self.n_groups,
                    group_size,
                    str(routing_weights.device),
                )
                routing_weights = routing_weights.unflatten(
                    self.dim, (self.n_groups, group_size)
                )
                topk_weights, topk_ids = torch.topk(
                    routing_weights, per_group_top_k, dim=-1
                )
                topk_ids = (topk_ids + group_offsets).flatten(-2, -1)
                topk_weights = topk_weights.flatten(-2, -1)
                return topk_weights, topk_ids
            else:
                routing_weights, selected_experts = moe_gating_topk_softmax(
                    x, self.top_k
                )
                return routing_weights, selected_experts

    dlinfer_moe.DlinferSoftmaxTopKImpl = PatchedDlinferSoftmaxTopKImpl


##### patch dlinfer rotary_embedding: DlinferRotaryEmbeddingBuilder.build #####
def patch_dlinfer_rotary_embedding():
    from lmdeploy.pytorch.backends.dlinfer import rotary_embedding as dlinfer_rotary
    from lmdeploy.pytorch.backends.rotary_embedding import (
        FopeParameters,
        Llama3Parameters,
        LongRoPEScalingParameters,
        RopeType,
        YarnParameters,
    )
    from lmdeploy.pytorch.backends.dlinfer.rotary_embedding import (
        DlinferLlama3RotaryEmbeddingImpl,
        DlinferLlamaDynamicNTKScalingRotaryEmbedding,
        DlinferRotaryEmbeddingImpl,
        DlinferYarnRotaryEmbeddingImpl,
    )

    @staticmethod
    def patched_build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        yarn_params: YarnParameters = None,
        longrope_params: LongRoPEScalingParameters = None,
        llama3_params: Llama3Parameters = None,
        fope_params: FopeParameters = None,
        emb_type: RopeType = RopeType.Default,
    ):
        """build."""
        if emb_type in (RopeType.Default, RopeType.LinearScaling):
            return DlinferRotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == RopeType.DynamicNTKScaling:
            return DlinferLlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings
            )
        elif emb_type == RopeType.Llama3:
            return DlinferLlama3RotaryEmbeddingImpl(
                dim,
                base,
                scaling_factor,
                llama3_params.low_freq_factor,
                llama3_params.high_freq_factor,
                max_position_embeddings,
            )
        elif emb_type == RopeType.Yarn:
            return DlinferYarnRotaryEmbeddingImpl(
                dim,
                base,
                scaling_factor,
                max_position_embeddings,
                yarn_params=yarn_params,
            )
        elif emb_type == RopeType.Fope:
            from lmdeploy.pytorch.backends.default.rotary_embedding import (
                FopeRotaryEmbeddingImpl,
            )

            return FopeRotaryEmbeddingImpl(
                dim,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
                params=fope_params,
            )
        else:
            raise NotImplementedError(f"Unsupported embedding type: {emb_type}")

    dlinfer_rotary.DlinferRotaryEmbeddingBuilder.build = patched_build


##### patch cache engine #####
def patch_contiguous_cache_engine():
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from functools import reduce
    from math import gcd
    from lmdeploy.pytorch.engine import cache_engine

    @classmethod
    def _cache_engine_allocate_caches(
        cls,
        num_blocks: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        world_size: int,
        device: str,
    ):
        """Allocate caches."""
        num_layers = model_config.num_layers

        # get all descs
        k_cache_desc = cls.get_k_cache_desc(model_config, cache_config, world_size)
        v_cache_desc = cls.get_v_cache_desc(model_config, cache_config, world_size)
        quant_cache_descs = cls.get_quant_cache_descs(
            k_cache_desc, v_cache_desc, model_config, cache_config
        )
        custom_cache_descs = cls.get_custom_cache_descs(model_config, cache_config)
        cache_descs = (
            [k_cache_desc, v_cache_desc] + quant_cache_descs + custom_cache_descs
        )

        # get mempool size
        mem_pool_size = 0
        alignments = []
        for desc in cache_descs:
            mem_pool_size += desc.aligned_size
            alignments.append(desc.alignment)

        # compute gcd of alignments
        alignments_gcd = reduce(gcd, alignments) if alignments else 1
        assert (
            mem_pool_size % alignments_gcd == 0
        ), "mem_pool_size must be divisible by alignments_gcd"

        # create pool
        mem_pool = torch.zeros(
            (mem_pool_size // alignments_gcd, num_layers, num_blocks, alignments_gcd),
            dtype=torch.uint8,
            device=device,
        )

        # slice caches
        caches = []
        remain_pool = mem_pool
        for desc in cache_descs:
            cache = (
                remain_pool[: desc.size // alignments_gcd, :, :, :]
                .view(desc.dtype)
                .view((num_layers, num_blocks, *desc.shape))
            )
            remain_pool = remain_pool[desc.aligned_size // alignments_gcd :, :, :, :]
            caches.append(cache)
        return mem_pool, caches

    cache_engine.CacheEngine.allocate_caches = _cache_engine_allocate_caches


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def vendor_device_init():
    import_vendor_module(vendor_name)
    patch_compiled_func()
    patch_async_sampling_logits()
    if vendor_name in ["camb", "ascend"]:
        patch_contiguous_cache_engine()
    patch_dlinfer_moe()
    patch_dlinfer_rotary_embedding()


vendor_device_init()
