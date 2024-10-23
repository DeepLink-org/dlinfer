# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
import dlinfer.ops as ext_ops


def InternAttention_naive_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
    q, k, v = qkv.unbind(0)
    if self.qk_normalization:
        q = self.q_norm(q)
        k = self.k_norm(k)

    attention_mask = None
    start_loc = torch.tensor(
        [N * i for i in range(B)], device=q.device, dtype=torch.int64
    )
    seq_len = torch.tensor([N for _ in range(B)], device=q.device, dtype=torch.int64)
    attn_output = ext_ops.prefill_attention(
        q,
        k,
        v,
        start_loc,
        seq_len,
        N,
        self.num_heads,
        self.num_heads,
        attention_mask,
        attn_output=x,
    )

    x = self.proj(attn_output.reshape(B, N, C))
    x = self.proj_drop(x)
    return x


def InternRMSNorm_forward(self, hidden_states):
    return ext_ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)
