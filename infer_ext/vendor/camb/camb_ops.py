import math
import torch
import torch_mlu
from bangtransformer.torch import bt_ops

from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

__all__ =[
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "get_cache_len",
]

@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float
) -> Tensor:
    assert 1 < hidden_states.ndim < 4, "only support hidden_states: [total_seq_len, head_size], [batch_size, seq_lens, head_size]"
    
    hidden_states = hidden_states.contiguous()
    shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, shape[-1])
    store_output_before_norm = False
    normed_hidden_states = bt_ops.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm)[0]
    normed_hidden_states = normed_hidden_states.view(shape)
    return normed_hidden_states

@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_full: Optional[Tensor],
    sin_full: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    assert query.ndim == 3, "only support q:[totalSeq, head ,head_dim]"
    assert key.ndim == 3, "only support k:[totalSeq, head ,head_dim]"
    interleaved = False
    embeded_query = torch.empty_like(query)
    embeded_key = torch.empty_like(key)
    if position_ids is not None:
        cos = cos_full[position_ids]
        sin = sin_full[position_ids]
    #view totalSeq as a long sequence
    cu_seq_lens = torch.Tensor([0,query.shape[0]]).long().mlu()
    max_context_len = query.shape[0]
    bt_ops.apply_rotary(embeded_query, query, sin, cos, position_ids, cu_seq_lens, interleaved, False, False, max_context_len)
    bt_ops.apply_rotary(embeded_key, key, sin, cos, position_ids, cu_seq_lens, interleaved, False, False, max_context_len)
    return embeded_query,embeded_key

if __name__ == '__main__':
    pass
