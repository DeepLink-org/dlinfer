import torch
from typing import List, Callable
from lmdeploy.pytorch.models import qwen2_vl


def dlinfer_apply_mrope_selection(
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


qwen2_vl._apply_mrope_selection = dlinfer_apply_mrope_selection
