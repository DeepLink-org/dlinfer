# Copyright (c) 2026, DeepLink. All rights reserved.

from types import SimpleNamespace

import torch

from dlinfer.framework.lmdeploy_ext.device import patch_contiguous_cache_engine
from lmdeploy.pytorch.engine.cache_engine import CacheDesc, CacheEngine


def test_contiguous_allocator_handles_cache_smaller_than_alignment(monkeypatch):
    k_desc = CacheDesc([128], torch.bfloat16)
    v_desc = CacheDesc([128], torch.bfloat16)
    index_desc = CacheDesc([128], torch.bfloat16)
    dummy_scale_desc = CacheDesc([1], torch.float32)

    monkeypatch.setattr(
        CacheEngine,
        "get_k_cache_desc",
        classmethod(lambda cls, *args, **kwargs: k_desc),
    )
    monkeypatch.setattr(
        CacheEngine,
        "get_v_cache_desc",
        classmethod(lambda cls, *args, **kwargs: v_desc),
    )
    monkeypatch.setattr(
        CacheEngine,
        "get_quant_cache_descs",
        classmethod(lambda cls, *args, **kwargs: []),
    )
    monkeypatch.setattr(
        CacheEngine,
        "get_custom_cache_descs",
        classmethod(lambda cls, *args, **kwargs: [index_desc, dummy_scale_desc]),
    )

    original_allocate = CacheEngine.__dict__["allocate_caches"]
    patch_contiguous_cache_engine()
    try:
        mem_pool, caches = CacheEngine.allocate_caches(
            num_blocks=3,
            model_config=SimpleNamespace(num_layers=2),
            cache_config=SimpleNamespace(),
            world_size=1,
            device="cpu",
        )
    finally:
        CacheEngine.allocate_caches = original_allocate

    assert mem_pool.shape[-1] == torch.float32.itemsize
    assert [cache.shape for cache in caches] == [
        torch.Size([2, 3, 128]),
        torch.Size([2, 3, 128]),
        torch.Size([2, 3, 128]),
        torch.Size([2, 3, 1]),
    ]
    assert [cache.dtype for cache in caches] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    ]
