import torch
import torch_mlu
from bangtransformer.torch import bt_ops
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

import sys 
sys.path.append("..") 
import camb_ops
# this is reference calculation
class ApplyRotary(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def rotate(self, x: torch.Tensor, interleaved: bool):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            y = torch.empty_like(x)
            x1, x2 = x[..., ::2], x[..., 1::2]
            y[..., ::2], y[..., 1::2] = -x2, x1
            return y

    def forward(self,
                output: torch.Tensor,       # [total_seqlen, num_heads, head_size]
                input: torch.Tensor,        # [total_seqlen, num_heads, head_size]
                sin_cache: torch.Tensor,    # [rope_seqlen, rotary_dim] / [batch, rope_seqlen, rotary_dim]
                cos_cache: torch.Tensor,    #
                position_id: torch.Tensor,  # [batch] / [batch, max_seqlen]
                cu_seqlen: torch.Tensor,    # [batch + 1]
                interleaved: bool,
                discrete: bool,
                dynamic: bool,
                max_seqlen: int):
        packed = input.dim() == 3
        rope_dim = sin_cache.shape[-1]
        batch_size = cu_seqlen.shape[0] - 1 if packed else input.shape[0]

        if position_id is not None and cu_seqlen is not None:
            if not discrete:
                tmp = []
                for i in range(batch_size):
                    seq = cu_seqlen[i + 1] - cu_seqlen[i]
                    tmp.extend([*range(position_id[i].item(), position_id[i].item() + seq)])
                position_id = torch.Tensor(tmp)

            sin_cache = sin_cache[position_id]
            cos_cache = cos_cache[position_id]

        for i in range(batch_size):
            input_i = input[cu_seqlen[i] : cu_seqlen[i + 1]] if packed else input[i]
            ouput_i = output[cu_seqlen[i] : cu_seqlen[i + 1]] if packed else output[i]
            input_i = input_i[..., 0:rope_dim]
            ouput_i = ouput_i[..., 0:rope_dim]
            sin_cache_i = sin_cache[i] if dynamic else sin_cache
            cos_cache_i = cos_cache[i] if dynamic else cos_cache
            seq = input_i.shape[0]
            sin_cache_i = sin_cache_i[:seq]
            cos_cache_i = cos_cache_i[:seq]
            rot = self.rotate(input_i, interleaved)

            ouput_i[:] = rot * sin_cache_i.unsqueeze(1) + input_i * cos_cache_i.unsqueeze(1)
        output[..., rope_dim:] = input[..., rope_dim:]

def test_apply_rotary_0():
    dtype=torch.half
    total_seq_len = 48
    q_heads = 16
    k_heads = 16
    head_dim = 32
    rope_dim = 2
    max_context_len = total_seq_len
    interleaved = False
    cu_seq_lens = torch.arange(0, 2, dtype=torch.int32).mlu() * total_seq_len

    query = torch.randn(size=(total_seq_len,q_heads,head_dim), dtype=dtype).mlu()
    key = torch.randn(size=(total_seq_len,k_heads,head_dim), dtype=dtype).mlu()
    cos_cache = torch.randn(size=(max_context_len, rope_dim), dtype=dtype).mlu()
    sin_cache = torch.randn(size=(max_context_len, rope_dim), dtype=dtype).mlu()

    ref_q_out = torch.empty_like(query)
    q_out = torch.empty_like(query)
    ref_k_out = torch.empty_like(key)
    k_out = torch.empty_like(key)

    apply_rotary = ApplyRotary()
    apply_rotary(ref_q_out, query.float(), sin_cache.float(), cos_cache.float(), \
        None, cu_seq_lens, interleaved, False, False, total_seq_len)
    apply_rotary(ref_k_out, key.float(), sin_cache.float(), cos_cache.float(), \
        None, cu_seq_lens, interleaved, False, False, total_seq_len)
    
    q_out, k_out = vendor_ops_registry["apply_rotary_pos_emb"](query, key, cos_cache, sin_cache, None, None, None)
    diff = (ref_q_out - q_out).mean()
    print("the mean difference between ref and dev is: ", diff)
    if diff < 0.0001:
        print("test_apply_rotary_0: pass")
    else:
        print("test_apply_rotary_0: not close")

if __name__ == '__main__':
    test_apply_rotary_0()


