# Copyright (c) 2024, DeepLink. All rights reserved.
import os
import math
import warnings
import torch
import torch_npu

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from .utils import SocVersion
from dlinfer.graph import config as graph_config

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
]


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
    # 注释掉调试打印，避免影响精度
    # print(f'####### in eager add_rms_norm!!!', flush=True)
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

    # 注释掉调试打印，避免影响精度
    # print(f'####### in eager apply_rotary_pos_emb!!!', flush=True)

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
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    if SocVersion.is_Ascend910():
        seq_qlen_list = (
            [max_q_seq_len * (i + 1) for i in range(query.shape[0])]
            if q_seq_len is None
            else q_seq_len.cumsum(0).tolist()
        )
        seq_kvlen_list = seq_qlen_list
        if (attn_mask is None or len(attn_mask) == 0) and q_seq_len is None:
            query = query.view(query.shape[0] * query.shape[1], num_q_heads, -1)
            key = key.view(key.shape[0] * key.shape[1], num_kv_heads, -1)
            value = value.view(value.shape[0] * value.shape[1], num_kv_heads, -1)
        # some vl models pass a fp16 mask from lmdeploy in vision part of prefill phase.
        attn_mask_ = (
            None
            if (attn_mask is None or len(attn_mask) == 0)
            else attn_mask[0].to(torch.bool)
        )
        attn_output.view(query.shape)[:] = torch.ops.npu.npu_fusion_attention(
            query,
            key,
            value,
            num_q_heads,
            "TND",
            scale=scale_value,
            atten_mask=attn_mask_,
            actual_seq_qlen=seq_qlen_list,
            actual_seq_kvlen=seq_kvlen_list,
        )[0]
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

    # 注释掉调试打印，避免影响精度
    # print(f'####### in eager fill_kv_cache!!!', flush=True)
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

    bs, _, dim = query.shape
    block_num = key_cache.size(0)
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

    # Check if we're in graph mode
    use_piecewise_graph = getattr(graph_config, "piecewise_graph_enabled", False)

    if not use_piecewise_graph:
        query = query.contiguous()
        query = query.view(bs, 1, num_q_heads * dim)
        key_cache = key_cache.view(block_num, block_size, -1)
        value_cache = value_cache.view(block_num, block_size, -1)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key_cache,
            value_cache,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=kv_seq_len,
            dequant_scale1=None,
            quant_scale1=None,
            dequant_scale2=None,
            quant_scale2=None,
            quant_offset2=None,
            antiquant_scale=kv_scales,
            antiquant_offset=kv_zeros,
            block_table=block_table,
            query_padding_size=None,
            kv_padding_size=None,
            key_antiquant_scale=None,
            key_antiquant_offset=None,
            value_antiquant_scale=None,
            value_antiquant_offset=None,
            key_shared_prefix=None,
            value_shared_prefix=None,
            actual_shared_prefix_len=None,
            query_rope=None,
            key_rope=None,
            key_rope_antiquant_scale=None,
            num_heads=num_q_heads,
            scale=scale_value,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="BSH",
            num_key_value_heads=num_kv_heads,
            sparse_mode=0,
            inner_precise=1,
            block_size=block_size,
            antiquant_mode=0,
            softmax_lse_flag=False,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
        )
        return attn_output

    # Replay mode - use _npu_paged_attention
    # Prepare tensors
    query = query.contiguous()

    # Ensure attn_output is not None and is contiguous
    if attn_output is None:
        raise RuntimeError("attn_output must be provided in graph mode")
    # attn_output = attn_output.contiguous()

    # Direct call to _npu_paged_attention without workspace for eager execution
    # import pdb;pdb.set_trace()
    # print(f'########### in replay paged_decode_attention, use torch_npu._npu_paged_attention!!!', flush=True)
    torch_npu._npu_paged_attention(
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

    return attn_output.view(bs, 1, num_q_heads * dim)


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

    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous().view(query.shape[0], 1, -1)
    block_num = key_cache.size(0)
    key_cache = key_cache.view(block_num, block_size, -1)
    value_cache = value_cache.view(block_num, block_size, -1)

    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query,
        key_cache,
        value_cache,
        pse_shift=None,
        atten_mask=attn_mask[0],
        actual_seq_lengths=kv_seq_len_list,
        actual_seq_lengths_kv=kv_seq_len,
        dequant_scale1=None,
        quant_scale1=None,
        dequant_scale2=None,
        quant_scale2=None,
        quant_offset2=None,
        antiquant_scale=kv_scales,
        antiquant_offset=kv_zeros,
        block_table=block_table,
        query_padding_size=None,
        kv_padding_size=None,
        key_antiquant_scale=None,
        key_antiquant_offset=None,
        value_antiquant_scale=None,
        value_antiquant_offset=None,
        key_shared_prefix=None,
        value_shared_prefix=None,
        actual_shared_prefix_len=None,
        query_rope=None,
        key_rope=None,
        key_rope_antiquant_scale=None,
        num_heads=num_q_heads,
        scale=scale_value,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BSH",
        num_key_value_heads=num_kv_heads,
        sparse_mode=0,
        inner_precise=1,
        block_size=block_size,
        antiquant_mode=0,
        softmax_lse_flag=False,
        key_antiquant_mode=0,
        value_antiquant_mode=0,
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


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
) -> Tensor:
    num_experts = gate_up_weights.size(0)
    active_num = hidden_states.size(0) * topk
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    if os.getenv("DLINFER_RESET_MOE_UPDATE_WEIGHTS", "0") == "1":
        gate_up_weights = gate_up_weights.transpose(1, 2)
        down_weights = down_weights.transpose(1, 2)

    # moe init routing
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

    # up sample
    group_list = expert_tokens.to(torch.int64)
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [expanded_hidden_states],
        [gate_up_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
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
        group_list_type=1,
    )[0]

    # moe finalize routing
    moe_output = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=down_proj, sorted_indices=expanded_row_idx, probs=topk_weights
    )

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
