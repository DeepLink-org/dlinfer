import math
import torch
import torch_mlu
import torch_mlu_ops as tmo

seqL = 8
head_num_q = 32
head_num_k = 16
head_size = 128
block_size = 16 
block_num = 2
dtype=torch.bfloat16
softmax_scale = 1.

#得到两句完全一样的query
query0 = torch.randn(1, 1, head_num_q, head_size, dtype=dtype).mlu()
query = query0.repeat(2,1,1,1)
print("we hare same query:", (query[0] == query[1]).all())

#得到完全一样的key，value的cache
key_cache0 = torch.randn(1, head_num_k, block_size, head_size, dtype=dtype).mlu()
value_cache0 = torch.randn(1, head_num_k, block_size, head_size, dtype=dtype).mlu()
key_cache = key_cache0.repeat(block_num,1,1,1)
value_cache = value_cache0.repeat(block_num,1,1,1)
# key_random = torch.randn(block_num,head_num_k,8,head_size, dtype=dtype).mlu()
# value_random = torch.randn(block_num,head_num_k,8,head_size, dtype=dtype).mlu()
# key_cache[:,:,(block_size- 8):,:] = key_random
# value_cache[:,:,(block_size- 8):,:] = value_random

for i in range(seqL):
    print(all(key_cache[0,:,i].flatten() == key_cache[1,:,i].flatten()))
    print(all(value_cache[0,:,i].flatten() == value_cache[1,:,i].flatten()))

#block_table
block_table = torch.tensor([[0],[1]], dtype=torch.int32).mlu()
print("block_table:",block_table)

#context_lens & max_kv_seq_len
context_lens = torch.tensor([seqL, seqL], dtype=torch.int32).mlu()
max_kv_seq_len = seqL
print("context_lens:",context_lens)
print("max_kv_seq_len:",max_kv_seq_len)

attn_output = torch.empty_like(query,dtype=dtype).mlu()

tmo.single_query_cached_kv_attn(query, key_cache, value_cache, attn_output, block_table, context_lens, None, None, \
    None, max_kv_seq_len, 0, 0, softmax_scale)

#q和kv相同->期望获得相同attn_output
print("we hare same attn_output:", (attn_output[0] == attn_output[1]).all())