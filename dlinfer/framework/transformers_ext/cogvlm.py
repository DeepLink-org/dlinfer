import torch
from torch import nn
import dlinfer.ops as ext_ops


def PatchedAttention_forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
    B, L, H = x.shape
    qkv = self.query_key_value(x)
    qkv = qkv.reshape(B, L, 3, H).permute(2, 0, 1, 3)  # 3, B, L, H
    q, k, v = qkv[0], qkv[1], qkv[2]

    out = ext_ops.prefill_attention(
        q,
        k,
        v,
        None,
        None,
        L,
        self.num_heads,
        self.num_heads,
        [],
        attn_output=q,
    )
    output = self.dense(out.view(B, L, -1))
    return output
