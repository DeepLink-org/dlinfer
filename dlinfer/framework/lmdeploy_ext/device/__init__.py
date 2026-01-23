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


vendor_device_init()
