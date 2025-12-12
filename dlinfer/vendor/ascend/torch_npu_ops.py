# Copyright (c) 2024, DeepLink. All rights reserved.
import os
import math
import torch
import torch.distributed as dist

from typing import List
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from .utils import SocVersion, get_vl_mask, get_cpu_seq_len
from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import (
    AscendGraphRunner,
    get_graph_params,
)

__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "incre_flash_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "get_cache_len",
    "weight_quant_matmul",
    "fused_moe",
    "linear",
    "rms_norm_w8a8",
    "add_rms_norm_w8a8",
    "dynamic_quant",
    "linear_w8a8",
]


@register_ops(vendor_ops_registry)
def rms_norm_w8a8(
    hidden_states: Tensor, weight: Tensor, epsilon: float, quant_dtype: torch.dtype
) -> Tuple[Tensor, Tensor]:
    hidden_states = hidden_states.contiguous()
    output = torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]
    x, scale = torch.ops.npu.npu_dynamic_quant(output, dst_type=quant_dtype)
    return x, scale


@register_ops(vendor_ops_registry)
def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:
    hidden_states = hidden_states.contiguous()
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    x, scale = torch.ops.npu.npu_dynamic_quant(
        normed_hidden_states, dst_type=quant_dtype
    )
    return x, scale, added_hidden_states


@register_ops(vendor_ops_registry)
def dynamic_quant(
    hidden_states: Tensor, quant_dtype: torch.dtype, quant_granularity: str
) -> Tuple[Tensor, Tensor]:
    assert quant_granularity == "PER_TOKEN"
    x, scale = torch.ops.npu.npu_dynamic_quant(hidden_states, dst_type=quant_dtype)
    return x, scale


@register_ops(vendor_ops_registry)
def linear_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    rms_scale: torch.Tensor,
    linear_scale: torch.Tensor,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias: Tensor,
) -> Tensor:

    out_dtype = torch.bfloat16 if out_dtype == torch.float16 else out_dtype
    hidden_states = hidden_states.squeeze(0)
    linear_scale = linear_scale.squeeze()
    rms_scale = rms_scale.squeeze(0)

    output = torch.ops.npu.npu_quant_matmul(
        hidden_states,
        weight.t(),
        linear_scale,
        pertoken_scale=rms_scale,
        bias=bias,
        output_dtype=out_dtype,
    )
    output = output.unsqueeze(0)
    return output


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    return normed_hidden_states, added_hidden_states


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    # rotary pos emb helpers:
    query = query.contiguous().unsqueeze(0)
    key = key.contiguous().unsqueeze(0)
    assert len(query.shape) == 4
    batch, seq_len, _, _ = query.shape
    cos = cos.reshape(batch, seq_len, 1, -1)
    sin = sin.reshape(batch, seq_len, 1, -1)

    def rotate_half_(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_(q, k, cos, sin):
        return (q * cos) + (rotate_half_(q) * sin), (k * cos) + (rotate_half_(k) * sin)

    # ascend ops currently only support dim 128
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        return apply_rotary_pos_emb_(query, key, cos, sin)
    return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")


@register_ops(vendor_ops_registry)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    if len(attn_mask):
        mask = attn_mask[0]
    else:
        mask = get_vl_mask(max_q_seq_len, query.dtype)
        q_seq_len = get_cpu_seq_len(q_seq_len)
    if SocVersion.is_Ascend910():
        torch.ops.atb._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=q_seq_len,
            scale_value=scale_value,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            out=attn_output,
        )
    elif SocVersion.is_Ascend310P():
        # Used for Qwen2.5-VL model vision block
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        attn_output[:] = torch.ops.npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BSND",
            scale_value=scale_value,
        )
    else:
        raise ValueError(
            f"dlinfer doesn't support {SocVersion.device_name()} device currently."
        )
    return attn_output


@register_ops(vendor_ops_registry)
def incre_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    input_layout: str,
    softmax_scale: Optional[float],
) -> Tensor:
    attn_output = torch.ops.npu.npu_incre_flash_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout=input_layout,
        scale_value=softmax_scale,
    )
    return attn_output


# atb._npu_reshape_and_cache has a performace advantage of about 3% compared to npu.npu_scatter_nd_update_,
# but atb._npu_reshape_and_cache will report an error when slot_indices is an empty tensor.
"""
@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> Tuple[Tensor, Tensor]:
    _, head, dim = key.shape
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()
    kv_indices = kv_indices.view(-1, 1)

    if quant_bits == 8:

        def quant_int8(x, x_scale, x_offset):
            quantized = (
                ((x / x_scale) - x_offset).round().clamp(-128, 127).to(torch.int8)
            )
            return quantized

        key = quant_int8(key, k_scales_zeros[0], k_scales_zeros[1])
        value = quant_int8(value, v_scales_zeros[0], v_scales_zeros[1])

    is_mla = key.shape[-1] != value.shape[-1]
    if is_mla:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    else:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        value_cache_reshaped = value_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
        torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache
"""


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> Tuple[Tensor, Tensor]:
    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    if quant_bits == 8:

        def quant_int8(x, x_scale, x_offset):
            quantized = (
                ((x / x_scale) - x_offset).round().clamp(-128, 127).to(torch.int8)
            )
            return quantized

        key = quant_int8(key, k_scales_zeros[0], k_scales_zeros[1])
        value = quant_int8(value, v_scales_zeros[0], v_scales_zeros[1])

    is_mla = key.shape[-1] != value.shape[-1]
    if is_mla:
        assert len(key_cache.shape) == 4
        key_cache_reshaped = torch.flatten(key_cache, start_dim=0, end_dim=1)
        kv_indices = kv_indices.view(-1, 1)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    else:
        torch.ops.atb._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=kv_indices.to(torch.int32),
        )
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    key_cache = torch.cat([key_cache, key_state], dim=1)
    value_cache = torch.cat([value_cache, value_state], dim=1)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def get_cache_len(cache: Tensor):
    return cache.shape[1]


@register_ops(vendor_ops_registry)
def decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_kv_heads: int,
    num_q_heads: int,
    scale_value: float,
    block_table: Tensor,
    kv_seq_len: Tensor,
    attn_output: Tensor,
):
    if AscendGraphRunner.capturing:
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_q_heads,
                scale_value,
                block_table,
                kv_seq_len,
                attn_output,
            )
        )
        graph_params.is_mla = False
        torch.npu.graph_task_group_begin(stream)
        torch.ops.atb._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            out=attn_output,
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
    else:
        torch.ops.atb._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            out=attn_output,
        )
    return attn_output


@register_ops(vendor_ops_registry)
def decode_attention_mla(
    query: Tensor,
    key_cache: Tensor,
    num_kv_heads: int,
    num_q_heads: int,
    scale_value: float,
    block_table: Tensor,
    kv_seq_len: Tensor,
    mla_vheadsize: int,
    attn_output: Tensor,
):
    if AscendGraphRunner.capturing:
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                query,
                key_cache,
                num_kv_heads,
                num_q_heads,
                scale_value,
                block_table,
                kv_seq_len,
                mla_vheadsize,
                attn_output,
            )
        )
        graph_params.is_mla = True
        torch.npu.graph_task_group_begin(stream)
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=mla_vheadsize,
            out=attn_output,
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
    else:
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=mla_vheadsize,
            out=attn_output,
        )
    return attn_output


@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    query = query.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    key_headsize, value_headsize = key_cache.shape[-1], value_cache.shape[-1]
    if key_headsize == value_headsize:
        return decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            kv_seq_len=kv_seq_len,
            attn_output=attn_output,
        )
    else:
        return decode_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            kv_seq_len=kv_seq_len,
            mla_vheadsize=value_headsize,
            attn_output=attn_output,
        )


@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    cu_seq_lens_kv: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )

    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous().view(query.shape[0], 1, -1)
    block_num = key_cache.size(0)
    key_cache = key_cache.view(block_num, block_size, -1)
    value_cache = value_cache.view(block_num, block_size, -1)

    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache,
        value=value_cache,
        atten_mask=attn_mask[0],
        block_table=block_table,
        input_layout="BSH",
        block_size=block_size,
        actual_seq_lengths=q_seq_len,
        actual_seq_lengths_kv=kv_seq_len,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale_value,
        sparse_mode=0,
    )

    return attn_output


@register_ops(vendor_ops_registry)
def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    hidden_states = hidden_states.contiguous()
    return torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


@register_ops(vendor_ops_registry)
def silu_and_mul(input_tensor: Tensor, dim: int) -> Tensor:
    if SocVersion.is_Ascend910():
        return torch.ops.npu.npu_swiglu(input_tensor, dim)
    elif SocVersion.is_Ascend310P():
        gate_cache, up_cache = input_tensor.chunk(2, dim)
        return torch.ops.npu.npu_silu(gate_cache) * up_cache


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights, selected_idx, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
        router_logits, None, topk
    )
    return routing_weights, selected_idx.to(torch.int64)


# TODO only for internlm in transformers lib.
# see issue #9 for details
@register_ops(vendor_ops_registry)
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: Sequence[Optional[Tensor]],
) -> Tensor:
    batch_size = query_states.shape[0]
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    q_seq_len, num_q_heads, _ = query_states.shape
    kv_seq_len, num_kv_heads, _ = value_states.shape
    attn_output = torch.empty_like(query_states)

    for i in range(batch_size):
        if q_seq_len == kv_seq_len:
            # mask must be a square
            if not mask[i : i + 1][0].shape[-1] == mask[i : i + 1][0].shape[-2]:
                min_shape = min(
                    mask[i : i + 1][0].shape[-1], mask[i : i + 1][0].shape[-2]
                )
                square_mask = mask[i : i + 1][0][..., :min_shape, :min_shape]
                square_mask = square_mask.contiguous()
            else:
                square_mask = mask[i : i + 1][0]

            prefill_attention(
                query_states,
                key_states,
                value_states,
                torch.tensor(
                    [kv_seq_len - q_seq_len],
                    dtype=torch.int64,
                    device=query_states.device,
                ),
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                q_seq_len,
                num_q_heads,
                num_kv_heads,
                [
                    square_mask,
                ],
                None,
                None,
                attn_output,
            )
        else:
            paged_decode_attention(
                query_states,
                key_states,
                value_states,
                None,
                0,
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                kv_seq_len,
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output


# Quantification of W4A16 is currently supported and tested.
@register_ops(vendor_ops_registry)
def weight_quant_matmul(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = False,
    group_size: Optional[int] = 0,
) -> Tensor:
    offset = None if (offset is None or offset.numel() == 0) else offset
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        qweight,
        scale,
        antiquant_offset=offset,
        antiquant_group_size=group_size,
        bias=bias,
    )


'''
def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=group
        )
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def token_dispatch(
    hidden_states: Tensor,
    topk_ids: Tensor,
    topk: int,
    num_experts: int,
    active_num: int,
    ep_size: int,
    ep_rank: int,
    ep_group: torch.distributed.ProcessGroup,
    expert_list: List[int],
):
    if ep_size <= 1:
        expanded_hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=active_num,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        return expanded_hidden_states, expanded_row_idx, expert_tokens
    else:
        num_tokens = hidden_states.size(0)
        global_num_experts = num_experts
        local_num_experts = len(expert_list)
        row_idx_len = active_num
        row_idx = (
            torch.arange(
                0, row_idx_len, dtype=torch.int32, device=torch.npu.current_device()
            )
            .view(topk, -1)
            .permute(1, 0)
            .contiguous()
        )
        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )

        global_expert_tokens = torch.bincount(
            expanded_expert_idx, minlength=global_num_experts
        )
        scatter_sizes = global_expert_tokens.view(ep_size, -1).sum(-1)

        gather_sizes = torch.empty_like(scatter_sizes)
        dist.all_to_all_single(gather_sizes, scatter_sizes, group=ep_group.device_group)
        scatter_size_list = scatter_sizes.cpu().tolist()
        gather_size_list = gather_sizes.cpu().tolist()

        expanded_expert_idx = expanded_expert_idx % local_num_experts
        hidden_states = ep_group.all_to_all(
            hidden_states, 0, 0, scatter_size_list, gather_size_list
        )
        local_expert_idx = ep_group.all_to_all(
            expanded_expert_idx, 0, 0, scatter_size_list, gather_size_list
        )

        sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)

        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            sorted_local_expert_idx, local_num_experts
        ).to(torch.int64)

        hidden_states = hidden_states[sorted_idx]
        # # token_dispatch preprocess
        # num_local_tokens_per_expert = torch.histc(
        #     topk_ids, bins=num_experts, min=0, max=num_experts
        # )
        # num_out_tokens = topk_ids.numel()
        # num_local_experts = len(expert_list)
        # input_splits = (
        #     num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
        #     .sum(axis=1)
        #     .to(torch.device("cpu"), non_blocking=True)
        #     .numpy()
        # )
        # num_global_tokens_per_expert = gather_from_sequence_parallel_region(
        #     num_local_tokens_per_expert, group=ep_group
        # ).reshape(ep_size, num_experts)
        # num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        #     :, expert_list[0] : expert_list[-1] + 1
        # ]
        # if num_global_tokens_per_local_expert is None:
        #     raise ValueError(
        #         "num_global_tokens_per_local_expert must be set before sum.")
        # output_splits = (num_global_tokens_per_local_expert.sum(
        #     axis=-1).to(torch.device("cpu"), non_blocking=True).numpy())
        # num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
        #     axis=0)
        # # ===================================================
        # # num_global_tokens_per_expert: [ep_size, num_experts]
        # # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # # num_tokens_per_local_expert: [num_local_experts]
        # # ===================================================

        # if num_local_experts > 1:
        #     if num_global_tokens_per_local_expert is None:
        #         raise ValueError(
        #             "num_global_tokens_per_local_expert must be set before operations."
        #         )
        #     global_input_tokens_local_experts_indices = torch.repeat_interleave(
        #         expert_ids_per_ep_rank,
        #         num_global_tokens_per_local_expert.ravel())
        # else:
        #     # TODO: This full synchronization can be a performance bottleneck.
        #     # A more granular sync (e.g., blocking D2H copies) should be investigated.
        #     torch.npu.synchronize()

        # # all_to_all
        # ...
        # # token_dispatch postprocess
        # ...


def token_combine(
    permuted_tokens: Tensor, sorted_indices: Tensor, topk_weights: Tensor, ep_size: int
):
    if ep_size <= 1:
        moe_output = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=permuted_tokens,
            sorted_indices=sorted_indices,
            probs=topk_weights,
        )
        return moe_output
    else:
        # token_combine preprocess
        ...
        # all_to_all
        ...
        # token_combine postprocess
        ...
'''


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    ep_size: int,
    ep_group: torch.distributed.ProcessGroup = None,
    expert_list: List[int] = None,
) -> Tensor:
    num_experts = gate_up_weights.size(0)
    active_num = hidden_states.size(0) * topk
    num_tokens = hidden_states.size(0)
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    if os.getenv("DLINFER_RESET_MOE_UPDATE_WEIGHTS", "0") == "1":
        gate_up_weights = gate_up_weights.transpose(1, 2)
        down_weights = down_weights.transpose(1, 2)

    # moe init routing
    if ep_size <= 1:
        expanded_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=active_num,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        group_list_type = 1
    else:
        quant_mode = 0
        moe_expert_num = ep_size * num_experts
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
            "expert_token_nums_type": 0,
        }
        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": ep_group.name(),
            "ep_world_size": ep_size,
            "ep_rank_id": ep_group.rank(),
        }
        kwargs_mc2.update(stage1_kwargs)
        distributed_moe_init_outputs = torch.ops.npu.npu_moe_distribute_dispatch(
            **kwargs_mc2
        )
        (
            expanded_hidden_states,
            dynamic_scale,
            assist_info_for_combine,
            expert_tokens,
            ep_recv_counts,
            _,
            expand_scales,
        ) = distributed_moe_init_outputs[0:7]
        group_list_type = 0

    # up sample
    group_list = expert_tokens.to(torch.int64)
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [expanded_hidden_states],
        [gate_up_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]

    # activation
    gate_cache = silu_and_mul(up_proj, -1)

    # down sample
    down_proj = torch.ops.npu.npu_grouped_matmul(
        [gate_cache],
        [down_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]

    # moe finalize routing
    if ep_size <= 1:
        moe_output = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=down_proj,
            sorted_indices=expanded_row_idx,
            probs=topk_weights,
        )
    else:
        moe_expert_num = ep_size * num_experts
        kwargs_mc2 = {
            "expand_x": down_proj,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
        }
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": ep_group.name(),
            "ep_world_size": ep_size,
            "ep_rank_id": ep_group.rank(),
            "expand_scales": expand_scales,
            "expand_idx": assist_info_for_combine,
        }
        kwargs_mc2.update(stage3_kwargs)
        moe_output = torch.ops.npu.npu_moe_distribute_combine(**kwargs_mc2)

    return moe_output


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    if all_reduce:
        assert group is None or group == "", "In eager mode, only use default_pg"
        group = torch.distributed.distributed_c10d._world.default_pg
        hcomm_info = group._get_backend(x.device).get_hccl_comm_name(x.device.index)
        out = torch.ops.npu.npu_mm_all_reduce_base(
            x.contiguous(),
            weight.transpose(0, 1),
            hcomm_info,
            reduce_op="sum",
            bias=bias,
        )
    else:
        # on 310p, the weight is transposed to nz format in llm part on graph mode,
        # but in vl part, eager mode is used.
        # we need to reshape it back to nd.
        if (
            len(weight.shape) == 4
            and weight.shape[0] == 1
            and weight.shape[1] * weight.shape[3] == x.shape[-1]
        ):
            weight = weight.permute(0, 2, 1, 3)
            weight = weight.reshape(weight.shape[1], -1)
        out = torch.nn.functional.linear(x, weight, bias)
    return out


@register_ops(vendor_ops_registry)
def transdata(
    hidden_states: Tensor,
    transdata_type: int,
):
    raise NotImplementedError("transdata in eager mode is not implemented yet!")
