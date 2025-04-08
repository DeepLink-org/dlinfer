import torch
from typing import List, Callable, Tuple, Optional, Any
from lmdeploy.pytorch.models import qwen2_vl
from lmdeploy.pytorch.models import deepseek_v2
from lmdeploy.pytorch.distributed import get_world_rank


# patch slice_scatter
def _apply_mrope_selection(
    hidden_states: torch.Tensor,
    mrope_position_ids: torch.Tensor,
    mrope_section: List[int],
    position_ids: torch.Tensor,
    rotary_emb_func: Callable,
):
    _mrope_position_ids = torch.clone(mrope_position_ids)
    cos, sin = rotary_emb_func(hidden_states, _mrope_position_ids)

    mrope_section = mrope_section * 2

    def _apply_split(src):
        start = 0
        dst_list = []
        for i, m in enumerate(src.split(mrope_section, dim=-1)):
            dst_list.append(m[i % 3])
            start += mrope_section[i]
        return torch.cat(dst_list, dim=-1)

    _cos = _apply_split(cos)
    _sin = _apply_split(sin)

    return _cos, _sin


# torch==2.3.1 dynamo does not support 'out' parameter in bmm
def DeepseekV2BMM_forward(self, x: torch.Tensor, output: torch.Tensor):
    out = torch.bmm(x.transpose(0, 1), self.weight).transpose(0, 1)
    output.copy_(out)


# replace slice_scatter with cat
def DeepseekV2Attention__kv_proj(self, hidden_states, nope_size: int):
    """kv proj."""
    # (q_len, 1, nope_size + pe_size)
    key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
    # (q_len, 1, pe_size)
    k_pe = key_states[..., nope_size:]
    # kv_a_layernorm
    value_states = key_states[..., :nope_size]
    value_states = self.kv_a_layernorm(value_states)
    key_states = torch.cat((value_states, k_pe), dim=-1)
    return key_states, value_states, k_pe


# replace slice_scatter with cat
def DeepseekV2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attn_metadata: Any = None,
):
    """Rewrite of LlamaAttention.forward."""
    world_size, _ = get_world_rank()
    num_heads = self.num_heads // world_size
    nope_size = self.kv_lora_rank
    q_len = hidden_states.size(1)

    # qkv_proj
    query_states, key_states, value_states, q_pe, k_pe = self._qkv_proj(
        hidden_states, num_heads=num_heads
    )

    cos, sin = rotary_pos_emb
    q_pe, k_pe = self.apply_rotary_pos_emb(
        q_pe,
        k_pe,
        cos,
        sin,
        inplace=False,
    )
    query_states = torch.cat((query_states[..., :nope_size], q_pe), dim=-1)
    key_states = torch.cat((key_states[..., :nope_size], k_pe), dim=-1)

    attn_output = self.attn_fwd(
        query_states,
        key_states,
        value_states,
        past_key_value[0],
        past_key_value[1],
        attn_metadata,
        k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
        v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
        inplace=True,
    )
    attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

    self.vc(attn_output, attn_bmm_out)
    attn_output = attn_bmm_out.flatten(-2, -1)[None]
    attn_output = self.o_proj(attn_output)

    return attn_output


qwen2_vl._apply_mrope_selection = _apply_mrope_selection
deepseek_v2.DeepseekV2BMM.forward = DeepseekV2BMM_forward
deepseek_v2.DeepseekV2Attention._kv_proj = DeepseekV2Attention__kv_proj
deepseek_v2.DeepseekV2Attention.forward = DeepseekV2Attention_forward
