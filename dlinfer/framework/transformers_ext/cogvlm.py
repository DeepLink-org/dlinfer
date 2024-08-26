import torch
from torch import nn
import dlinfer.ops as ext_ops


def PatchedAttentionForward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
    B, L, H = x.shape
    qkv = self.query_key_value(x)
    qkv = qkv.reshape(B, L, 3, H).permute(2, 0, 1, 3)  # 3, B, L, H
    q, k, v = qkv[0], qkv[1], qkv[2]

    ext_ops.prefill_attention(
        q, k, v, None, None, L, self.num_heads, self.num_heads, None, attn_output=q
    )
    output = self.dense(q.view(B, L, -1))
    return output
