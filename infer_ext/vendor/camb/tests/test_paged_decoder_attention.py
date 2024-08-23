import torch
import torch_mlu
import random
import math
from bangtransformer.torch import bt_ops
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

import sys
sys.path.append("..")
import camb_ops

class SingleQueryAttn(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        is_pagedattn: paged_attn or single_query_cachedKV_attn
    """
    def __init__(self, softmax_scale, is_pagedattn = True):
        super().__init__()
        self.is_pagedattn = is_pagedattn
        self.softmax_scale = softmax_scale

    def masked_attention(self,
                      query,
                      key,
                      value,
                      alibi_slope,
                      context_len,
                      qk_scale
    ) -> torch.Tensor:
         # (num_heads, seq_q, seq_k)
        qk = torch.einsum('qhd,hkd->hqk', query, key)
        qk = qk * qk_scale

        if alibi_slope is not None:
            alibi_dist = torch.arange(0, context_len, dtype=torch.float32).mlu()
            alibi = alibi_slope[:, None] * alibi_dist
            qk = qk + alibi[:, None, :]
            # num_heads = query.size(1)
            # block_size=key.size(-2)
            # alibi = self.build_alibi(alibi_slope, block_size, num_heads, dtype=torch.float32)
            # qk = qk + alibi[..., -1, :].view(num_heads, 1, block_size)

        _, seq_q, seq_k = qk.size()
        if seq_q > 1:
            #add triu mask
            ml = torch.zeros((seq_q, seq_k - seq_q), dtype=qk.dtype).mlu()
            ones = torch.ones((seq_q, seq_q), dtype=qk.dtype).mlu() * -torch.inf
            mr = torch.triu(ones, diagonal=1)
            mask = torch.cat((ml, mr), dim=-1)
            qk = qk + mask
        attention = torch.softmax(qk, dim = -1, dtype=qk.dtype)
        qkv = torch.einsum('hqk,hkd->qhd', attention, value)
        return qkv
    def forward(self, query,
                key_cache, value_cache, block_tables, context_lens,
                key_scale, value_scale, alibi_slopes):
        if key_scale is not None:
           key_cache *= key_scale.reshape(*key_scale.shape, 1)
        if value_scale is not None:
           value_cache *= value_scale.reshape(*key_scale.shape, 1)
        bs, seq_q, num_heads, head_size = query.size()
        num_blocks, num_kv_heads, block_size, _ = key_cache.size()
        output = torch.zeros((bs, seq_q, num_heads, head_size), dtype=torch.float16)

        assert (num_heads % num_kv_heads == 0)
        head_repeats = num_heads // num_kv_heads
        for bs_id in range(bs):
            q_bs = query[bs_id]
            block_table = block_tables[bs_id]
            context_len = int(context_lens[bs_id])

            table_end = (context_len + block_size - 1) // block_size
            block_ids = block_table[0 : table_end]
            keys, values = key_cache[block_ids], value_cache[block_ids]

            keys = torch.repeat_interleave(keys, head_repeats, dim=1)
            keys = keys.transpose(1, 0).contiguous().view(num_heads, -1, head_size)
            keys = keys[:, 0:context_len, :]

            values = torch.repeat_interleave(values, head_repeats, dim=1)
            values = values.transpose(1, 0).contiguous().view(num_heads, -1, head_size)
            values = values[:, 0:context_len, :]

            alibi_slope = alibi_slopes[bs_id] if alibi_slopes is not None else None
            qkv= self.masked_attention(q_bs, keys, values, alibi_slope, context_len, self.softmax_scale)
            output[bs_id] = qkv
        return output



def test_ref_paged_attn_0():
    dtype=torch.half
    batch_size = 32
    num_blocks = 4
    block_size= 16
    head_num = 16
    num_kv_heads = 16
    max_seqlen = 64
    head_size = 64
    is_pagedattn = True
    kv_data_type = torch.float16
    has_alibi = False
    query = torch.randn(size=(batch_size, 1, head_num,head_size), dtype=dtype).mlu()
    context_lens = torch.randint(1, max_seqlen + 1, (batch_size, ), dtype=torch.int32).mlu()
    max_context_len = int(max(context_lens))
    if is_pagedattn:
        block_size = 16
    else:
        block_size = max_seqlen
    num_blocks = (int)(batch_size * (max_seqlen / block_size))
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    cache_shape = (num_blocks, num_kv_heads, block_size, head_size)
    scale_shape = (num_blocks, num_kv_heads, block_size)
    block_tables = random.sample(range(0, num_blocks), batch_size * max_num_blocks_per_seq)
    block_tables = torch.tensor(block_tables, dtype=torch.int32).mlu().view(batch_size, max_num_blocks_per_seq)
    if kv_data_type is not torch.int8:
        key_cache = torch.randn(size=cache_shape, dtype=torch.float16).mlu()
        value_cache = torch.randn(size=cache_shape, dtype=torch.float16).mlu()
        key_cache_scale = None
        value_cache_scale = None
    else:
        key_cache = torch.zeros(cache_shape).uniform_(-128, 128).to(kv_data_type).mlu()
        value_cache = torch.zeros(cache_shape).uniform_(-128, 128).to(kv_data_type).mlu()
        key_cache_scale = torch.randn(size=scale_shape, dtype=torch.float32).mlu()
        value_cache_scale = torch.randn(size=scale_shape, dtype=torch.float32).mlu()

    alibi_slopes = None
    if has_alibi:
        alibi_slopes = torch.zeros((batch, head_num), dtype=torch.float32).mlu()
        alibi_slopes.uniform_(0, 0.125)
            
    softmax_scale = 1 / math.sqrt(head_size)
    attention = SingleQueryAttn(softmax_scale=softmax_scale, is_pagedattn = is_pagedattn)
    ref_output = attention(query.contiguous().float(),
                                     key_cache.float(),
                                     value_cache.float(),
                                     block_tables,
                                     context_lens,
                                     key_cache_scale,
                                     value_cache_scale,
                                     alibi_slopes)
    dev_output = torch.zeros_like(query).mlu()
    dev_out = vendor_ops_registry["paged_decode_attention"](query, key_cache, value_cache, block_tables, block_size, context_lens, head_num, num_kv_heads, None, None, dev_output)
    diff = (ref_output - dev_output.cpu()).mean()
    print("the mean difference between ref and dev is: ", diff)
    if diff and torch.allclose(ref_output, dev_output.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False) < 0.0001:
        print("test_apply_rotary_0: pass")
    else:
        print("test_apply_rotary_0: not close")
if __name__ == '__main__':
    test_ref_paged_attn_0()
