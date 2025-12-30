import torch
import numpy
from abc import ABC, abstractmethod
from torch import Tensor
from dlinfer.utils.type_annotation import DlinferDistContext


def fused_moe_mc2(): ...


def fused_moe_all2all(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    dist_ctx: DlinferDistContext,
):
    num_local_experts = gate_up_weights.size(0)
    num_experts = num_local_experts * dist_ctx.ep_size

    input_splits = None
    output_splits = None
    hidden_shape_before_permute = None
    num_global_tokens_per_local_expert = None
    reversed_global_input_permutation_mapping = None
    reversed_local_input_permutation_mapping = None
    global_input_tokens_local_experts_indices = None

    def dispatch(hidden_states, topk_ids):
        nonlocal input_splits
        nonlocal output_splits
        nonlocal hidden_shape_before_permute
        nonlocal num_global_tokens_per_local_expert
        nonlocal reversed_global_input_permutation_mapping
        nonlocal reversed_local_input_permutation_mapping
        nonlocal global_input_tokens_local_experts_indices
        # dispatch pre-process
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=num_experts, min=0, max=num_experts
        )
        num_out_tokens = topk_ids.numel()
        input_splits = (
            num_local_tokens_per_expert.reshape(dist_ctx.ep_size, num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = torch.empty(
            (num_local_tokens_per_expert.size(0) * dist_ctx.ep_size,),
            dtype=num_local_tokens_per_expert.dtype,
            device=torch.npu.current_device(),
        )
        torch.distributed.all_gather_into_tensor(
            num_global_tokens_per_expert, num_local_tokens_per_expert, dist_ctx.ep_group
        )
        num_global_tokens_per_expert = num_global_tokens_per_expert.reshape(
            dist_ctx.ep_size, num_experts
        )
        hidden_shape_before_permute = hidden_states.shape
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :,
            num_local_experts
            * dist_ctx.ep_rank : num_local_experts
            * (dist_ctx.ep_rank + 1),
        ]
        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
        if dist_ctx.ep_size > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            expert_ids_per_ep_rank = torch.tensor(
                [i % num_local_experts for i in range(num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
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
        global_input_tokens = permutated_local_input_tokens.new_empty(
            size=[sum(output_splits)] + list(permutated_local_input_tokens.size()[1:]),
            dtype=permutated_local_input_tokens.dtype,
            device=torch.npu.current_device(),
        )
        # dispatch
        permute1_ep_all_to_all_handle = torch.distributed.all_to_all_single(
            global_input_tokens,
            permutated_local_input_tokens,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=dist_ctx.ep_group,
            async_op=True,
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # dispatch post-process
        if num_local_experts <= 1:
            return global_input_tokens, None

        global_input_tokens, reversed_global_input_permutation_mapping = (
            torch.ops.npu.npu_moe_token_permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )
        )
        return {
            "hidden_states": global_input_tokens,
            "group_list": num_tokens_per_local_expert,
            "dynamic_scale": None,
            "group_list_type": 1,
        }

    def combine(hidden_states, gate_up_weights, topk_weights):
        nonlocal input_splits
        nonlocal output_splits
        nonlocal hidden_shape_before_permute
        nonlocal num_global_tokens_per_local_expert
        nonlocal reversed_global_input_permutation_mapping
        nonlocal reversed_local_input_permutation_mapping
        nonlocal global_input_tokens_local_experts_indices
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
            group=dist_ctx.ep_group,
            async_op=True,
        )
        unpermute1_ep_all_to_all_handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=gloabl_output_tokens,
            sorted_indices=reversed_local_input_permutation_mapping.to(torch.int32),
            probs=topk_weights,
            restore_shape=hidden_shape_before_permute,
        )
        input_splits = None
        output_splits = None
        hidden_shape_before_permute = None
        num_global_tokens_per_local_expert = None
        reversed_global_input_permutation_mapping = None
        reversed_local_input_permutation_mapping = None
        global_input_tokens_local_experts_indices = None
        return output

    def fused_moe_all2all_forward(
        hidden_states: torch.Tensor,
        gate_up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()
        dispatched_outputs = dispatch(hidden_states, topk_ids)
        # up sample
        group_list = dispatched_outputs["group_list"].to(torch.int64)
        up_proj = torch.ops.npu.npu_grouped_matmul(
            [dispatched_outputs["hidden_states"]],
            [gate_up_weights],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=dispatched_outputs["group_list_type"],
        )[0]

        # activation
        gate_cache = torch.ops.npu.npu_swiglu(up_proj, -1)

        # down sample
        down_proj = torch.ops.npu.npu_grouped_matmul(
            [gate_cache],
            [down_weights],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=dispatched_outputs["group_list_type"],
        )[0]
        combined_output = combine(down_proj, gate_up_weights, topk_weights)
        return combined_output

    return fused_moe_all2all_forward(
        hidden_states, gate_up_weights, down_weights, topk_ids, topk_weights
    )
