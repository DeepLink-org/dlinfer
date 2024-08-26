import torch
import torch_mlu
import random
from bangtransformer.torch import bt_ops
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from torch.nn.parameter import Parameter

import sys 
sys.path.append("..") 
import camb_ops

class ReshapePagedCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k: torch.Tensor, v: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor,
                slot_mapping: torch.Tensor):
        num_tokens = k.shape[0]
        block_size = k_cache.shape[2]
        for i in range(num_tokens):
            if slot_mapping[i] >= 0:
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode='floor')
                block_offset = slot_mapping[i] % block_size
                k_cache[block_id, :, block_offset, :] = k[i]
                v_cache[block_id, :, block_offset, :] = v[i]

def test_fill_kv_cache_0():
    test_cases = 10

    num_tokens_list = torch.randint(low=1, high=1024, size=(test_cases, ), dtype=torch.int32)
    num_heads_list = torch.randint(low=1, high=64, size=(test_cases, ), dtype=torch.int32)
    head_size_list = torch.randint(low=1, high=16, size=(test_cases, ), dtype=torch.int32)
    head_size_list *= 16
    block_size_list = torch.randint(low=1, high=4, size=(test_cases, ), dtype=torch.int32)
    block_size_list *= 16
    
    for i in range(test_cases):

        num_tokens = num_tokens_list[i]
        num_heads = num_heads_list[i]
        head_size = head_size_list[i]
        block_size = block_size_list[i]
    
        min_blocks = (int)((num_tokens + block_size - 1) / block_size)
        num_blocks = min(min_blocks + 10, 2 * min_blocks)
    
        print("num_tokens: {}, num_heads: {}, head_size: {}, num_blocks: {}, block_size: {}, testing...".format(
                    num_tokens, num_heads, head_size, num_blocks, block_size), flush=True)
        qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=torch.half).mlu()
        _, key, value = qkv.unbind(dim=1)
        key_cache   = torch.randn(num_blocks, num_heads, block_size, head_size, dtype=torch.half).mlu()
        value_cache = torch.randn(num_blocks, num_heads, block_size, head_size, dtype=torch.half).mlu()
    
        num_slots = num_blocks * block_size
        slot_mapping = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int).mlu()

        ref_key_cache, ref_value_cache = key_cache.clone(), value_cache.clone()
        reshape_paged_cache = ReshapePagedCache()

        reshape_paged_cache(key, value, ref_key_cache, ref_value_cache, slot_mapping)
        key_cache, value_cache = vendor_ops_registry["fill_kv_cache"](key, value, key_cache, value_cache, slot_mapping)

        key_cache = key_cache.cpu().float()
        ref_key_cache = ref_key_cache.cpu().float()
        value_cache = value_cache.cpu().float()
        ref_value_cache = ref_value_cache.cpu().float()
        allclose = torch.allclose(key_cache, ref_key_cache) and \
            torch.allclose(value_cache, ref_value_cache)
        max_diff = max(torch.max(torch.abs(key_cache - ref_key_cache)), 
            torch.max(torch.abs(value_cache - ref_value_cache)))
        if allclose and max_diff < 0.0001:
            print(f"test_fill_kv_cache case{i}: allclose, max diff: {max_diff}")
        else:
            print(f"test_fill_kv_cache case{i}: not close, max diff: {max_diff}")

if __name__ == "__main__":
    test_fill_kv_cache_0()
