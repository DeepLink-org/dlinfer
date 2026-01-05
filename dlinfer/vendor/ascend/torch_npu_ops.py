# Copyright (c) 2024, DeepLink. All rights reserved.
import os
import math
import torch
import torch.distributed as dist

from typing import List
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import (
    Tensor,
    Optional,
    Sequence,
    Tuple,
    DlinferDistContext,
)
from .utils import SocVersion, get_cpu_seq_len
from .attention import decode_attention, decode_attention_mla
from . import moe
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.backends.dlinfer.ascend.op_backend import AscendOpsBackend

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
        # Handle qwenvl vision part flash-attention
        q_seq_len = get_cpu_seq_len(q_seq_len)
        torch.ops.atb._npu_flash_attention_unpad(
            query=query,
            key=key,
            value=value,
            seq_len=q_seq_len,
            scale_value=scale_value,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            out=attn_output,
        )
        return attn_output
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
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            scale_value=scale_value,
            block_table=block_table,
            block_size=block_size,
            kv_seq_len=kv_seq_len,
            softmax_scale=softmax_scale,
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
def moe_gating_topk_softmax(
    router_logits: Tensor, topk: int, dist_ctx: DlinferDistContext
) -> Tuple[Tensor, Tensor]:
    if dist_ctx.ep_size > 1:
        paded_size = (
            (AscendOpsBackend.max_tokens_accros_dp + dist_ctx.tp_size - 1)
            // dist_ctx.tp_size
            * dist_ctx.tp_size
        )
        pad_size = paded_size - router_logits.shape[0]
        router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, pad_size))
        if dist_ctx.tp_size > 1:
            split_router_logits = torch.tensor_split(
                router_logits, dist_ctx.tp_size, dim=0
            )
            router_logits = split_router_logits[dist_ctx.tp_rank]
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


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    dist_ctx: DlinferDistContext,
) -> Tensor:
    hidden_states, split_hidden_states, num_tokens, x_active_mask, moe_group_name = (
        moe.moe_prepare(hidden_states, dist_ctx)
    )

    topk_ids = topk_ids.to(torch.int32)
    if os.getenv("DLINFER_RESET_MOE_UPDATE_WEIGHTS", "0") == "1":
        gate_up_weights = gate_up_weights.transpose(1, 2)
        down_weights = down_weights.transpose(1, 2)

    if dist_ctx.ep_size <= 1:
        moe_output = moe.fused_moe_tp(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            topk,
            renormalize,
        )

    elif AscendOpsBackend.max_tokens_accros_dp <= dist_ctx.tp_size * 512:
        moe_output = moe.fused_moe_mc2(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            topk,
            renormalize,
            dist_ctx,
            moe_group_name,
            x_active_mask,
        )
    else:
        moe_output = moe.fused_moe_all2all(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            topk,
            renormalize,
            dist_ctx,
        )

    moe_output = moe.moe_finalize(split_hidden_states, moe_output, num_tokens, dist_ctx)

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
