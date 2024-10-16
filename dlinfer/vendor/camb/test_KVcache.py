import math
import torch
import torch_mlu
import torch_mlu_ops as tmo

#tmo 提供的reference 
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

num_blocks = 2
num_heads = 2
block_size = 16
head_size = 2
dtype = torch.float
num_tokens = block_size
key = torch.ones(num_tokens,num_heads,head_size,dtype=dtype).mlu()
value = torch.ones(num_tokens,num_heads,head_size,dtype=dtype).mlu()
kv_indices = torch.arange(0, num_tokens).to(torch.int).mlu()

#tmo计算，不符合预期，key_cache在block_size维度有一半没有被赋值
key_cache   = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=dtype).mlu()
value_cache   = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=dtype).mlu()
print("before:",key_cache.mean())
print("kv_indices:",kv_indices.dtype)
tmo.reshape_paged_cache(key, value, key_cache, value_cache, kv_indices)
print("after:",key_cache.mean())
print("after:",key_cache[0,:,:,].mean())

#reference计算，结果符合预期
key_cache_ref   = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=dtype).mlu()
value_cache_ref   = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=dtype).mlu()
reshape_paged_cache = ReshapePagedCache()
reshape_paged_cache(key, value, key_cache_ref, value_cache_ref, kv_indices)
print("after:",key_cache_ref.mean())
print("after:",key_cache_ref[0,:,:,].mean())