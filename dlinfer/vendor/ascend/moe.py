import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from dlinfer.utils.type_annotation import MoECommType

# aclnnGroupedMatmulV5 requires the groupList tensor to have at most 1024
# entries. Models with more experts than this (e.g. meta-MoE with 2560
# experts) must split the grouped matmul into several sub-calls. The limit can
# be overridden via the DLINFER_MAX_GROUP_LIST_SIZE environment variable.
MAX_GROUP_LIST_SIZE = int(os.environ.get("DLINFER_MAX_GROUP_LIST_SIZE", "1024"))

# Prefill (non-capturing) MoE path for >1024 experts:
#   True  -> catch-all (identical scheme to graph capture)
#   False -> per-chunk row slicing (weight views, each row computed once)
_MOE_PREFILL_USE_CATCHALL = os.environ.get("DLINFER_MOE_PREFILL_CATCHALL", "0") == "1"


class ChunkedMoeWeightLayout:
    """Physical chunk layout for graph-mode chunked MoE weights."""

    def __init__(
        self,
        num_experts: int,
        chunk_size: int,
        packed: bool = False,
    ):
        self.num_experts = num_experts
        self.chunk_size = chunk_size
        self.packed = packed


def build_chunked_moe_storage_layout(num_experts: int):
    """Return packed storage size/layout for direct weight loading."""
    if num_experts <= MAX_GROUP_LIST_SIZE:
        return num_experts, None

    chunk_size = MAX_GROUP_LIST_SIZE - 2
    num_chunks = (num_experts + chunk_size - 1) // chunk_size
    storage_num_experts = num_experts + 2 * num_chunks
    return storage_num_experts, ChunkedMoeWeightLayout(
        num_experts=num_experts,
        chunk_size=chunk_size,
        packed=True,
    )


def chunked_moe_storage_expert_id(
    expert_id: int, chunked_moe_layout: ChunkedMoeWeightLayout
):
    """Map logical expert id to its packed catch-all storage id."""
    return expert_id + 2 * (expert_id // chunked_moe_layout.chunk_size) + 1


def zero_chunked_moe_weight_padding(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    chunked_moe_layout: ChunkedMoeWeightLayout,
):
    """Zero the leading/trailing catch-all experts in packed MoE weights."""
    start_expert = 0
    chunk_idx = 0
    while start_expert < chunked_moe_layout.num_experts:
        end_expert = min(
            start_expert + chunked_moe_layout.chunk_size,
            chunked_moe_layout.num_experts,
        )
        offset = start_expert + 2 * chunk_idx
        end_offset = offset + (end_expert - start_expert) + 1
        gate_up_weights[offset].zero_()
        gate_up_weights[end_offset].zero_()
        down_weights[offset].zero_()
        down_weights[end_offset].zero_()
        start_expert = end_expert
        chunk_idx += 1


_FUSED_MOE_OP_AVAILABLE = None
_FUSED_MOE_EXPERT_TOKEN_NUMS = {}


def _prepend_env_path(name: str, value: Path):
    """Make a vendor library/search path visible before loading the extension."""
    old_value = os.environ.get(name, "")
    value = str(value)
    if old_value:
        if value not in old_value.split(":"):
            os.environ[name] = f"{value}:{old_value}"
    else:
        os.environ[name] = value


def _find_vllm_ascend_extension():
    """Find the prebuilt vLLM-Ascend extension without importing vLLM."""
    candidates = []
    for env_name in (
        "DLINFER_ASCEND_CUSTOM_OP_LIBRARY",
        "VLLM_ASCEND_CUSTOM_OP_LIBRARY",
    ):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))

    # Source-tree location used by the Ascend development environment.
    candidates.extend(Path("/vllm-workspace/vllm-ascend/vllm_ascend").glob("vllm_ascend_C*.so"))
    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        candidates.extend((base / "vllm_ascend").glob("vllm_ascend_C*.so"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_fused_moe_op() -> bool:
    """Load the prebuilt dispatch_ffn_combine operator, if available."""
    global _FUSED_MOE_OP_AVAILABLE
    if _FUSED_MOE_OP_AVAILABLE is not None:
        return _FUSED_MOE_OP_AVAILABLE

    try:
        getattr(torch.ops._C_ascend, "dispatch_ffn_combine")
        _FUSED_MOE_OP_AVAILABLE = True
        return True
    except AttributeError:
        pass

    extension = _find_vllm_ascend_extension()
    if extension is None:
        _FUSED_MOE_OP_AVAILABLE = False
        return False

    vendor_path = extension.parent / "_cann_ops_custom/vendors/custom_transformer"
    if vendor_path.is_dir():
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)
        vendor_lib_path = vendor_path / "op_api/lib"
        if vendor_lib_path.is_dir():
            _prepend_env_path("LD_LIBRARY_PATH", vendor_lib_path)

    try:
        torch.ops.load_library(str(extension))
        getattr(torch.ops._C_ascend, "dispatch_ffn_combine")
    except (ImportError, OSError, RuntimeError, AttributeError):
        _FUSED_MOE_OP_AVAILABLE = False
    else:
        _FUSED_MOE_OP_AVAILABLE = True
    return _FUSED_MOE_OP_AVAILABLE


def fused_moe_mc2_w8a8(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_size: int,
    ep_rank: int,
    moe_group_name: str,
    x_active_mask: torch.Tensor,
    max_output_size: int = 512,
    swiglu_limit: float = 0.0,
):
    """Run the prebuilt Ascend W8A8 dispatch/MLP/combine operator."""
    if not ensure_fused_moe_op():
        raise RuntimeError("dispatch_ffn_combine is unavailable")

    num_local_experts = gate_up_weights.shape[0]
    cache_key = (hidden_states.device.type, hidden_states.device.index, num_local_experts)
    expert_token_nums = _FUSED_MOE_EXPERT_TOKEN_NUMS.get(cache_key)
    if expert_token_nums is None:
        expert_token_nums = torch.zeros(
            (num_local_experts,), dtype=torch.int32, device=hidden_states.device
        )
        _FUSED_MOE_EXPERT_TOKEN_NUMS[cache_key] = expert_token_nums

    # The operator accepts a list so that dynamic-EPLB can provide one tensor
    # per weight shard. GLM-5.2 uses one local shard per rank.
    empty_bias = torch.empty((0,), dtype=torch.float32)
    out = torch.empty_like(hidden_states)
    result, _ = torch.ops._C_ascend.dispatch_ffn_combine(
        x=hidden_states,
        weight1=[gate_up_weights],
        weight2=[down_weights],
        expert_idx=topk_ids.to(torch.int32),
        scale1=[gate_up_scales],
        scale2=[down_scales],
        bias1=[empty_bias],
        bias2=[empty_bias],
        probs=topk_weights.to(torch.float32),
        group=moe_group_name,
        max_output_size=int(max_output_size),
        swiglu_limit=float(swiglu_limit),
        world_size=int(ep_size),
        x_active_mask=x_active_mask,
        out=out,
        expert_token_nums=expert_token_nums,
    )
    return result


# ---------------------------------------------------------------------------
# mega_moe (cann-9.2.1 cann_ops_transformer): fused dispatch+MoE+combine for
# prefill. Each SymmBuffer owns an HCCL communicator, so ALL layers share one
# buffer per process; capacity grows monotonically in pow2 steps and every
# rank takes the same growth decision (padded rows are identical across
# ranks). Retired buffers are never destroyed: collective close is unverified
# here, and growth happens at most a handful of times.
# ---------------------------------------------------------------------------
_MEGA_SYMMS = []          # [{"sym": SymmBuffer, "capacity": int}]
_MEGA_EXPERT_LISTS = {}   # (w1.data_ptr(), w2.data_ptr()) -> (w1, w2, l1, l2, l1s, l2s)


def _get_mega_symm_buffer(ep_group, num_experts, hidden, intermediate_hidden,
                          topk, num_tokens):
    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe
    need = max(1, 1 << (num_tokens - 1).bit_length())
    if _MEGA_SYMMS and _MEGA_SYMMS[-1]["capacity"] >= need:
        return _MEGA_SYMMS[-1]["sym"]
    ep_world = torch.distributed.get_world_size(ep_group)
    sym = get_symm_buffer_for_mega_moe(
        ep_group, num_experts, need, topk,
        hidden=hidden, intermediate_hidden=intermediate_hidden,
        max_recv_token_num=min(65536, need * ep_world * min(topk, num_experts // ep_world)),
        dispatch_quant_mode=2,        # CANN_MEGA_MOE_QUANT_MODE_INT8
        dispatch_quant_out_dtype=258,  # CANN_ACL_INT8
    )
    _MEGA_SYMMS.append({"sym": sym, "capacity": need})
    return sym


def _mega_expert_lists(gate_up_weights, down_weights, gate_up_scales, down_scales):
    """Per-expert NZ weight / packed-scale lists, cached per weight identity.

    Entries are views (no device memory); one per layer. The `is` check guards
    against data_ptr reuse after update_weights swaps in new tensors."""
    key = (gate_up_weights.data_ptr(), down_weights.data_ptr())
    hit = _MEGA_EXPERT_LISTS.get(key)
    if hit is not None and hit[0] is gate_up_weights and hit[1] is down_weights:
        return hit[2:]
    if len(_MEGA_EXPERT_LISTS) > 512:
        _MEGA_EXPERT_LISTS.clear()
    e = gate_up_weights.shape[0]
    lists = (
        list(gate_up_weights.unbind(0)),
        list(down_weights.unbind(0)),
        list(gate_up_scales.view(e, -1).unbind(0)),
        list(down_scales.view(e, -1).unbind(0)),
    )
    _MEGA_EXPERT_LISTS[key] = (gate_up_weights, down_weights, *lists)
    return lists


def fused_moe_mega_w8a8(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_size: int,
    ep_group: torch.distributed.ProcessGroup,
    topk: int,
    x_active_mask: torch.Tensor,
    swiglu_limit: float = 0.0,
):
    """cann mega_moe: fused dispatch+MoE+combine (prefill).

    Same weight contract as fused_moe_mc2_w8a8 (int8 NZ + packed int64
    per-out-channel scales). Output = this rank's own tokens' full combine,
    so moe_finalize consumes it exactly like the MC2 result. Contract
    verified in /afs-weights/wangqing/deepep_bench/mega_moe_acc.py."""
    from cann_ops_transformer.ops import mega_moe
    if hidden_states.dtype != torch.bfloat16:  # mega always outputs bf16
        raise ValueError(f"MEGA MoE requires bf16 hidden states, got {hidden_states.dtype}")
    num_experts = ep_size * gate_up_weights.shape[0]
    sym = _get_mega_symm_buffer(ep_group, num_experts,
                                hidden_states.shape[1], gate_up_weights.shape[-1],
                                topk, hidden_states.shape[0])
    l1, l2, l1s, l2s = _mega_expert_lists(gate_up_weights, down_weights,
                                          gate_up_scales, down_scales)
    out, _ = mega_moe(
        hidden_states, topk_ids, topk_weights.to(torch.float32),
        l1, l2, sym,
        l1_weights_sf=l1s, l2_weights_sf=l2s,
        x_active_mask=x_active_mask.to(torch.int8),  # bool is rejected (EZ0020)
        weight1_type=258, weight2_type=258,          # ACL_INT8
        activation="swiglu",
        activation_clamp=float(swiglu_limit) or None,
    )
    return out


def _gmm_w8a8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return the (E, in, out) view of an expert weight for grouped matmul.

    ND weights are stored OI (E, out, in) and need a transpose; NZ weights
    (acl format 29, the dispatch_ffn_combine contract) are already IO.
    GLM-5.3's gate_up is square (2*2048 == hidden == 4096), so the layout
    must be read from the tensor format, not from the shape.
    """
    try:
        if torch_npu.get_npu_format(weight) == 29:
            return weight
    except (RuntimeError, AttributeError):
        pass
    return weight.transpose(1, 2)


def _grouped_mlp(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
):
    # up sample
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [hidden_states],
        [gate_up_weights.transpose(1, 2)],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]

    # activation
    gate_cache = torch.ops.npu.npu_swiglu(up_proj, -1)

    # down sample
    down_proj = torch.ops.npu.npu_grouped_matmul(
        [gate_cache],
        [down_weights.transpose(1, 2)],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]
    return down_proj


def _grouped_mlp_w8a8(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    input_scale: torch.Tensor = None,
    model_dtype: torch.dtype = None,
    swiglu_limit: float = 0.0,
):
    """Public-op W8A8 MoE fallback: quant GMM1, SwiGLU, quant GMM2.

    This intentionally uses only public NPU operators.  The intermediate GMM1
    output is dequantized to the model dtype, then requantized per token after
    SwiGLU, matching vllm-ascend's non-fused fallback.
    """
    if input_scale is None:
        quantized, input_scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.int8
        )
        gemm1_out_dtype = hidden_states.dtype
    else:
        # D8: activations arrived pre-quantized from the dispatch A2A
        quantized = hidden_states
        gemm1_out_dtype = model_dtype
    gate_up_weight = _gmm_w8a8_weight(gate_up_weights)
    gate_up = torch.ops.npu.npu_grouped_matmul(
        [quantized],
        [gate_up_weight],
        scale=[gate_up_scales.squeeze(-1)],
        per_token_scale=[input_scale],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=gemm1_out_dtype,
    )[0]

    # GLM-5.3 the clamped SwiGLU: gate at +limit, up at +-limit.  Unclamped
    # models (swiglu_limit=0) match the plain npu_swiglu path.
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit > 0:
        gate = gate.float().clamp_max(swiglu_limit)
        up = up.float().clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = (gate * torch.sigmoid(gate) * up).to(gate_up.dtype)
    activated_quant, activated_scale = torch.ops.npu.npu_dynamic_quant(
        activated, dst_type=torch.int8
    )
    down_weight = _gmm_w8a8_weight(down_weights)
    return torch.ops.npu.npu_grouped_matmul(
        [activated_quant],
        [down_weight],
        scale=[down_scales.squeeze(-1)],
        per_token_scale=[activated_scale],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=gemm1_out_dtype,  # D8: model dtype, x may be int8
    )[0]


def apply_mlp_w8a8(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    input_scale: torch.Tensor = None,
    model_dtype: torch.dtype = None,
    swiglu_limit: float = 0.0,
):
    """Apply W8A8 experts through the non-fused public-op path."""
    if gate_up_weights.size(0) > MAX_GROUP_LIST_SIZE:
        raise RuntimeError(
            "W8A8 fallback supports at most 1024 local experts; "
            f"got {gate_up_weights.size(0)}"
        )
    return _grouped_mlp_w8a8(
        hidden_states,
        gate_up_weights,
        gate_up_scales,
        down_weights,
        down_scales,
        group_list,
        group_list_type,
        input_scale=input_scale,
        model_dtype=model_dtype,
        swiglu_limit=swiglu_limit,
    )


def _apply_mlp_chunked_eager(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    token_counts: torch.Tensor,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
):
    """Eager / prefill path for >1024 experts.

    The expanded hidden_states rows are ordered by expert, so each chunk of
    experts owns a contiguous block of rows whose boundaries come from the
    cumulative token counts. This slices both weights and rows per chunk, which
    is the cheapest option but needs host-side row offsets (``.tolist()``) and
    is therefore only valid outside NPU graph capture.

    With packed layouts, the same single weight tensor is used and the two
    catch-all padding rows around each chunk are skipped.
    """
    num_experts = (
        chunked_moe_layout.num_experts
        if chunked_moe_layout is not None
        else gate_up_weights.size(0)
    )
    row_ends = torch.cumsum(token_counts, dim=0).tolist()

    outputs = []
    start_expert = 0
    start_row = 0
    chunk_idx = 0
    while start_expert < num_experts:
        if chunked_moe_layout is not None and chunked_moe_layout.packed:
            end_expert = min(start_expert + chunked_moe_layout.chunk_size, num_experts)
            storage_offset = start_expert + 2 * chunk_idx + 1
            real_len = end_expert - start_expert
            gate_up = gate_up_weights[storage_offset : storage_offset + real_len]
            down = down_weights[storage_offset : storage_offset + real_len]
        else:
            end_expert = min(start_expert + MAX_GROUP_LIST_SIZE, num_experts)
            gate_up = gate_up_weights[start_expert:end_expert]
            down = down_weights[start_expert:end_expert]
        end_row = row_ends[end_expert - 1]
        outputs.append(
            _grouped_mlp(
                hidden_states[start_row:end_row],
                gate_up,
                down,
                token_counts[start_expert:end_expert],
                1,
            )
        )
        start_expert = end_expert
        start_row = end_row
        chunk_idx += 1
    return torch.cat(outputs, dim=0)


def _apply_mlp_chunked_capturable(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    token_counts: torch.Tensor,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
):
    """NPU-graph-capturable path for >1024 experts.

    Host-side row slicing (``.tolist()``/``.cpu()``/``.item()``) triggers a
    synchronizing device->host copy, which is illegal during graph capture.
    Instead each chunk of experts is run over the *full* row set, with the
    chunk's real experts flanked by two zero-weight "catch-all" groups covering
    the rows of all other experts. Because ``swiglu(0) == 0`` and ``0 @ w == 0``
    those out-of-chunk rows produce exactly zero, so a given row is non-zero in
    exactly one chunk and the chunk outputs can simply be summed. Only static
    expert boundaries and device-side reductions are used, so no host sync
    happens and every tensor keeps a fixed shape across replays.

    Packed layouts use static weight slices. Direct calls without a packed
    layout still assemble grouped weights with ``torch.cat`` per chunk.
    """
    num_experts = gate_up_weights.size(0)
    if chunked_moe_layout is not None:
        num_experts = chunked_moe_layout.num_experts
    chunk_size = MAX_GROUP_LIST_SIZE - 2

    use_chunked_layout = (
        chunked_moe_layout is not None
        and chunked_moe_layout.num_experts == num_experts
        and chunked_moe_layout.chunk_size == chunk_size
        and chunked_moe_layout.packed
    )
    if not use_chunked_layout:
        zeros_up = gate_up_weights.new_zeros((1,) + tuple(gate_up_weights.shape[1:]))
        zeros_down = down_weights.new_zeros((1,) + tuple(down_weights.shape[1:]))

    output = None
    start_expert = 0
    chunk_idx = 0
    while start_expert < num_experts:
        end_expert = min(start_expert + chunk_size, num_experts)
        leading = token_counts[:start_expert].sum().reshape(1)
        trailing = token_counts[end_expert:].sum().reshape(1)
        group_list = torch.cat(
            [leading, token_counts[start_expert:end_expert], trailing]
        )
        if use_chunked_layout and chunked_moe_layout.packed:
            offset = start_expert + 2 * chunk_idx
            chunk_len = end_expert - start_expert + 2
            gate_up = gate_up_weights[offset : offset + chunk_len]
            down = down_weights[offset : offset + chunk_len]
        else:
            gate_up = torch.cat(
                [zeros_up, gate_up_weights[start_expert:end_expert], zeros_up], dim=0
            )
            down = torch.cat(
                [zeros_down, down_weights[start_expert:end_expert], zeros_down], dim=0
            )
        chunk_out = _grouped_mlp(hidden_states, gate_up, down, group_list, 1)
        output = chunk_out if output is None else output + chunk_out
        start_expert = end_expert
        chunk_idx += 1
    return output


def apply_mlp(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
):
    num_experts = (
        chunked_moe_layout.num_experts
        if chunked_moe_layout is not None
        else gate_up_weights.size(0)
    )
    if num_experts <= MAX_GROUP_LIST_SIZE:
        return _grouped_mlp(
            hidden_states, gate_up_weights, down_weights, group_list, group_list_type
        )

    # More experts than aclnnGroupedMatmulV5 supports: split into chunks of at
    # most MAX_GROUP_LIST_SIZE groups. Work in per-expert token counts
    # (group_list_type=1) regardless of the incoming layout.
    if group_list_type == 0:
        # group_list is a cumulative sum -> recover per-expert counts
        token_counts = group_list.clone()
        token_counts[1:] = group_list[1:] - group_list[:-1]
    else:
        token_counts = group_list

    # Graph capture cannot tolerate the host sync of the eager row-slicing path,
    # so capture always uses the catch-all path. Prefill keeps the cheaper eager
    # row-slicing path and can read real expert slices from packed weights.
    from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import (
        AscendGraphRunner,
    )

    if AscendGraphRunner.capturing or _MOE_PREFILL_USE_CATCHALL:
        return _apply_mlp_chunked_capturable(
            hidden_states,
            gate_up_weights,
            down_weights,
            token_counts,
            chunked_moe_layout,
        )
    return _apply_mlp_chunked_eager(
        hidden_states, gate_up_weights, down_weights, token_counts, chunked_moe_layout
    )


def moe_prepare(
    hidden_states: torch.Tensor,
    x_active_mask: torch.Tensor,
    pad_size: int,
    tp_size: int,
    ep_size: int,
    tp_rank: int,
    moe_comm_type: MoECommType,
    topk_ids: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
):
    if ep_size <= 1:
        return hidden_states, None, None, None, topk_ids, topk_weights
    # MEGA (cann mega_moe) shares MC2's pad/mask/TP-split contract
    mc2_like = moe_comm_type in (MoECommType.MC2, MoECommType.MEGA)
    num_tokens = hidden_states.size(0)
    # When tp_size > 1, topk_ids may need TP-splitting. Models using
    # moe_gating_topk_softmax already split there; detect by comparing
    # with the original (pre-pad) token count.
    need_topk_pad = topk_ids is not None and topk_ids.shape[0] == num_tokens
    need_topk_split = need_topk_pad and tp_size > 1
    # pad hidden_states (and topk tensors if needed)
    if pad_size > 0:
        if mc2_like:
            x_active_mask = torch.nn.functional.pad(
                x_active_mask, (0, pad_size), value=False
            )
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
        if need_topk_pad:
            topk_ids = torch.nn.functional.pad(topk_ids, (0, 0, 0, pad_size))
            topk_weights = torch.nn.functional.pad(topk_weights, (0, 0, 0, pad_size))
    if mc2_like and need_topk_pad:
        # dispatch_ffn_combine mis-computes a chunk that owns exactly ONE
        # active token (standalone rel ~5% at model dims; >=2 active and
        # fully-masked chunks are exact). Grow the global padding so every
        # TP chunk has >=2 rows, then duplicate a lone active row into the
        # neighbouring pad slot. All arithmetic is on host ints; the extra
        # rows are dropped by moe_finalize's [:num_tokens] slice.
        if hidden_states.size(0) < 2 * tp_size:
            extra = 2 * tp_size - hidden_states.size(0)
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, extra))
            topk_ids = torch.nn.functional.pad(topk_ids, (0, 0, 0, extra))
            topk_weights = torch.nn.functional.pad(topk_weights, (0, 0, 0, extra))
            if x_active_mask is not None:
                x_active_mask = torch.nn.functional.pad(
                    x_active_mask, (0, extra), value=False
                )
    # split hidden_states, x_active_mask, and topk tensors if tp_size > 1
    paded_num_tokens = hidden_states.size(0)
    if tp_size > 1:
        split_hidden_states = torch.tensor_split(hidden_states, tp_size, dim=0)
        hidden_states = split_hidden_states[tp_rank]
        if mc2_like:
            split_x_active_mask = torch.tensor_split(x_active_mask, tp_size, dim=0)
            x_active_mask = split_x_active_mask[tp_rank]
        if need_topk_split:
            topk_ids = torch.tensor_split(topk_ids, tp_size, dim=0)[tp_rank]
            topk_weights = torch.tensor_split(topk_weights, tp_size, dim=0)[tp_rank]
        if mc2_like and need_topk_pad:
            chunk = paded_num_tokens // tp_size
            if num_tokens == tp_rank * chunk + 1:
                # this rank's lone active row sits at chunk position 0
                hidden_states = hidden_states.clone()
                hidden_states[1] = hidden_states[0]
                topk_ids = topk_ids.clone()
                topk_ids[1] = topk_ids[0]
                topk_weights = topk_weights.clone()
                topk_weights[1] = topk_weights[0]
                x_active_mask = x_active_mask.clone()
                x_active_mask[1] = True
    return (
        hidden_states,
        num_tokens,
        paded_num_tokens,
        x_active_mask,
        topk_ids,
        topk_weights,
    )


def moe_finalize(
    moe_output: torch.Tensor,
    num_tokens: int,
    paded_num_tokens: int,
    ep_size: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
):
    if ep_size > 1:
        if tp_size > 1:
            output_shape = list(moe_output.shape)
            output_shape[0] = paded_num_tokens
            gathered_output = torch.empty(
                output_shape, dtype=moe_output.dtype, device=moe_output.device
            )
            split_hidden_states = torch.tensor_split(gathered_output, tp_size, dim=0)
            dist.all_gather(list(split_hidden_states), moe_output, tp_group)
            moe_output = gathered_output
        if moe_output.size(0) > num_tokens:
            moe_output = moe_output[:num_tokens, :].contiguous()
    return moe_output


def fused_moe_naive(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
    gate_up_scales: torch.Tensor = None,
    down_scales: torch.Tensor = None,
    swiglu_limit: float = 0.0,
):
    num_experts = (
        chunked_moe_layout.num_experts
        if chunked_moe_layout is not None
        else gate_up_weights.size(0)
    )
    active_num = hidden_states.size(0) * topk
    # do renormalize
    if renormalize:
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    # distribute dispatch
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

    # MLP
    group_list_type = 1
    expert_tokens = expert_tokens.to(torch.int64)
    if gate_up_scales is None:
        mlp_output = apply_mlp(
            expanded_hidden_states,
            gate_up_weights,
            down_weights,
            expert_tokens,
            group_list_type,
            chunked_moe_layout,
        )
    else:
        mlp_output = apply_mlp_w8a8(
            expanded_hidden_states,
            gate_up_weights,
            gate_up_scales,
            down_weights,
            down_scales,
            expert_tokens,
            group_list_type,
            swiglu_limit=swiglu_limit,
        )

    # distribute combine
    moe_output = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=mlp_output,
        sorted_indices=expanded_row_idx,
        probs=topk_weights,
    )
    return moe_output


def fused_moe_mc2(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    renormalize: bool,
    ep_size: int,
    ep_rank: int,
    moe_group_name: str,
    x_active_mask: torch.Tensor,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
    gate_up_scales: torch.Tensor = None,
    down_scales: torch.Tensor = None,
    swiglu_limit: float = 0.0,
):
    # do renormalize
    if renormalize:
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    # distribute dispatch
    num_local_experts = (
        chunked_moe_layout.num_experts
        if chunked_moe_layout is not None
        else gate_up_weights.size(0)
    )
    quant_mode = 0
    moe_expert_num = num_local_experts * ep_size
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
        "group_ep": moe_group_name,
        "ep_world_size": ep_size,
        "ep_rank_id": ep_rank,
    }
    stage1_kwargs.update(
        {
            "group_tp": moe_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }
    )
    stage1_kwargs.update(
        {
            "x_active_mask": x_active_mask,
        }
    )
    kwargs_mc2.update(stage1_kwargs)
    distributed_moe_init_outputs = torch.ops.npu.npu_moe_distribute_dispatch_v2(
        **kwargs_mc2
    )
    (
        expanded_hidden_states,
        dynamic_scale,
        assist_info_for_combine,
        expert_tokens,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = distributed_moe_init_outputs[0:7]

    # MLP
    group_list_type = 0
    if gate_up_scales is None:
        mlp_output = apply_mlp(
            expanded_hidden_states,
            gate_up_weights,
            down_weights,
            expert_tokens,
            group_list_type,
            chunked_moe_layout,
        )
    else:
        mlp_output = apply_mlp_w8a8(
            expanded_hidden_states,
            gate_up_weights,
            gate_up_scales,
            down_weights,
            down_scales,
            expert_tokens,
            group_list_type,
            swiglu_limit=swiglu_limit,
        )

    # distribute combine
    kwargs_mc2 = {
        "expand_x": mlp_output,
        "expert_ids": topk_ids,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
    }
    stage3_kwargs = {
        "ep_send_counts": ep_recv_counts,
        "group_ep": moe_group_name,
        "ep_world_size": ep_size,
        "ep_rank_id": ep_rank,
        "expand_scales": expand_scales,
        "assist_info_for_combine": assist_info_for_combine,
    }
    stage3_kwargs.update(
        {
            "tp_send_counts": tp_recv_counts,
            "group_tp": moe_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }
    )
    stage3_kwargs.update(
        {
            "x_active_mask": x_active_mask,
        }
    )
    kwargs_mc2.update(stage3_kwargs)
    moe_output = torch.ops.npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
    return moe_output


def fused_moe_all2all(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    renormalize: bool,
    ep_size: int,
    ep_rank: int,
    ep_group: dist.ProcessGroup,
    expert_ids_per_ep_rank: torch.Tensor,
    chunked_moe_layout: ChunkedMoeWeightLayout = None,
    gate_up_scales: torch.Tensor = None,
    down_scales: torch.Tensor = None,
    swiglu_limit: float = 0.0,
):
    num_local_experts = (
        chunked_moe_layout.num_experts
        if chunked_moe_layout is not None
        else gate_up_weights.size(0)
    )
    num_experts = num_local_experts * ep_size

    def dispatch(hidden_states, topk_ids):
        # dispatch pre-process
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=num_experts, min=0, max=num_experts
        )
        num_out_tokens = topk_ids.numel()
        hidden_shape_before_permute = hidden_states.shape

        num_global_tokens_per_expert = torch.empty(
            (ep_size, num_experts),
            dtype=num_local_tokens_per_expert.dtype,
            device=hidden_states.device,
        )
        torch.distributed.all_gather_into_tensor(
            num_global_tokens_per_expert.view(-1), num_local_tokens_per_expert, ep_group
        )

        local_splits_tensor = num_local_tokens_per_expert.view(
            ep_size, num_local_experts
        ).sum(dim=1)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, num_local_experts * ep_rank : num_local_experts * (ep_rank + 1)
        ]
        global_splits_tensor = num_global_tokens_per_local_expert.sum(dim=-1)

        combined_splits = torch.cat(
            [local_splits_tensor, global_splits_tensor]
        ).tolist()
        input_splits = combined_splits[:ep_size]
        output_splits = combined_splits[ep_size:]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
        if num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = (
            torch.ops.npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=topk_ids,
                num_out_tokens=num_out_tokens,
            )
        )

        # D8 (DLINFER_A2A_PREQUANT=1): quantize to int8 + per-token scale
        # BEFORE the A2A so the wire carries 1B/elt instead of 2B (vllm-ascend
        # All2All parity). Same values as quantizing after the exchange, so
        # the W8A8 GEMM output is bit-identical.
        prequant = (
            os.environ.get("DLINFER_A2A_PREQUANT", "0") == "1"
            and gate_up_scales is not None
        )
        local_scale = None
        if prequant:
            (
                permutated_local_input_tokens,
                local_scale,
            ) = torch.ops.npu.npu_dynamic_quant(
                permutated_local_input_tokens, dst_type=torch.int8
            )

        # dispatch
        global_input_tokens = permutated_local_input_tokens.new_empty(
            size=[sum(output_splits)] + list(permutated_local_input_tokens.size()[1:]),
            dtype=permutated_local_input_tokens.dtype,
            device=torch.npu.current_device(),
        )
        permute1_ep_all_to_all_handle = torch.distributed.all_to_all_single(
            global_input_tokens,
            permutated_local_input_tokens,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
            async_op=True,
        )
        permute1_ep_all_to_all_handle.wait()
        # permutated_local_input_tokens.untyped_storage().resize_(0)
        del permutated_local_input_tokens

        global_scale = None
        if prequant:
            global_scale = local_scale.new_empty(
                (sum(output_splits),) + local_scale.shape[1:]
            )
            scale_a2a_handle = torch.distributed.all_to_all_single(
                global_scale,
                local_scale,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=ep_group,
                async_op=True,
            )
            scale_a2a_handle.wait()
            del local_scale

        # dispatch post-process
        if num_local_experts <= 1:
            return global_input_tokens, None

        global_input_tokens, reversed_global_input_permutation_mapping = (
            torch.ops.npu.npu_moe_token_permute(
                # token_permute handles int8 tokens (standalone-verified);
                # init_routing_v2's joint scale reorder is subtly wrong here
                global_input_tokens, global_input_tokens_local_experts_indices
            )
        )
        if prequant:
            # mapping[k] = sorted position of original row k (verified:
            # out[mapping] == input; unpermute consumes it the same way).
            # Realign scale by scatter: s_sorted[mapping[k]] = s[k].
            m = reversed_global_input_permutation_mapping.to(torch.long)
            sorted_scale = torch.empty_like(global_scale)
            sorted_scale[m] = global_scale
            global_scale = sorted_scale

        context_metadata = {
            "hidden_shape_before_permute": hidden_shape_before_permute,
            "input_splits": input_splits,
            "output_splits": output_splits,
            "reversed_local_input_permutation_mapping": reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping": reversed_global_input_permutation_mapping,
        }

        return {
            "hidden_states": global_input_tokens,
            "dynamic_scale": global_scale,
            "group_list": num_tokens_per_local_expert,
            "group_list_type": 1,
            "context_metadata": context_metadata,
        }

    def combine(
        hidden_states,
        gate_up_weights,
        topk_weights,
        hidden_shape_before_permute,
        input_splits,
        output_splits,
        reversed_local_input_permutation_mapping,
        reversed_global_input_permutation_mapping,
    ):
        if hidden_states.shape[0] > 0 and gate_up_weights.shape[0] > 1:
            hidden_states = torch.ops.npu.npu_moe_token_unpermute(
                hidden_states, reversed_global_input_permutation_mapping
            )
        gloabl_output_tokens = hidden_states.new_empty(
            size=[sum(input_splits)] + list(hidden_states.size()[1:]),
            dtype=hidden_states.dtype,
            device=torch.npu.current_device(),
        )
        unpermute1_ep_all_to_all_handle = torch.distributed.all_to_all_single(
            gloabl_output_tokens,
            hidden_states,
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
            group=ep_group,
            async_op=True,
        )
        unpermute1_ep_all_to_all_handle.wait()
        # hidden_states.untyped_storage().resize_(0)
        del hidden_states

        output = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=gloabl_output_tokens,
            sorted_indices=reversed_local_input_permutation_mapping.to(torch.int32),
            probs=topk_weights,
            restore_shape=hidden_shape_before_permute,
        )
        output = output.view(hidden_shape_before_permute)
        return output

    def fused_moe_all2all_forward(
        hidden_states: torch.Tensor,
        gate_up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # do renormalize
        if renormalize:
            topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        # distribute dispatch
        dispatched_outputs = dispatch(hidden_states, topk_ids)

        # MLP
        group_list = dispatched_outputs["group_list"].to(torch.int64)
        if gate_up_scales is None:
            mlp_output = apply_mlp(
                dispatched_outputs["hidden_states"],
                gate_up_weights,
                down_weights,
                group_list,
                dispatched_outputs["group_list_type"],
                chunked_moe_layout,
            )
        else:
            mlp_output = apply_mlp_w8a8(
                dispatched_outputs["hidden_states"],
                gate_up_weights,
                gate_up_scales,
                down_weights,
                down_scales,
                group_list,
                dispatched_outputs["group_list_type"],
                input_scale=dispatched_outputs["dynamic_scale"],
                model_dtype=hidden_states.dtype,
                swiglu_limit=swiglu_limit,
            )

        # distribute combine
        context_metadata = dispatched_outputs["context_metadata"]
        combined_output = combine(
            mlp_output,
            gate_up_weights,
            topk_weights,
            context_metadata["hidden_shape_before_permute"],
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            context_metadata["reversed_local_input_permutation_mapping"],
            context_metadata["reversed_global_input_permutation_mapping"],
        )
        return combined_output

    return fused_moe_all2all_forward(
        hidden_states, gate_up_weights, down_weights, topk_ids, topk_weights
    )
