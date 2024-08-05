import infer_ext.ops as ext_ops
import torch
from torch import Tensor

def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    unsqueeze_dim=2,
    q_embed=None,
    k_embed=None,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    ext_ops.apply_rotary_pos_emb(query_states, key_states,
                                 cos, sin, None, None, None)
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    return q_embed, k_embed

def context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: list,
):
    batch_size = query_states.shape[0]
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    q_seq_len, num_q_heads, _ = query_states.shape
    kv_seq_len, num_kv_heads, _ = value_states.shape
    attn_output = torch.empty_like(query_states)

    for i in range(batch_size):
        if q_seq_len == kv_seq_len:
            ext_ops.context_attention(
                attn_output,
                query_states,
                key_states,
                value_states,
                torch.tensor([kv_seq_len-q_seq_len], dtype=torch.int64, device=query_states.device),
                torch.tensor([kv_seq_len], dtype=torch.int64, device=query_states.device),
                num_q_heads,
                num_kv_heads,
                mask[i:i + 1],
            )
        else:
            ext_ops.paged_decode_attention(
                attn_output,
                query_states,
                key_states,
                value_states,
                block_table=None,
                block_size=0,
                kv_seq_len=torch.tensor([kv_seq_len], dtype=torch.int64, device=query_states.device),
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads
            )
    return attn_output
