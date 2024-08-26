import torch
from torch import nn
import dlinfer.ops as ext_ops


class PatchedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, H = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, H).permute(2, 0, 1, 3)  # 3, B, L, H
        q, k, v = qkv[0], qkv[1], qkv[2]

        ext_ops.prefill_attention(
            q, k, v,
            None, None, L,
            self.num_heads, self.num_heads,
            None, attn_output=q
        )
        output = self.dense(q.view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

