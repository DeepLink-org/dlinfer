import torch
import numpy
import torch.distributed as dist


def apply_mlp(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
):
    # up sample
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [hidden_states],
        [gate_up_weights],
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
        [down_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]
    return down_proj


def moe_prepare(
    hidden_states: torch.Tensor,
    x_active_mask: torch.Tensor,
    pad_size: int,
    tp_size: int,
    ep_size: int,
    tp_rank: int,
    ep_group: dist.ProcessGroup,
):
    if ep_size <= 1:
        return hidden_states, None, None, None, None
    num_tokens = hidden_states.size(0)
    local_rank = torch.distributed.get_rank(group=ep_group)
    backend = ep_group._get_backend(torch.device("npu"))
    moe_group_name = backend.get_hccl_comm_name(local_rank)
    # pad hidden_states
    if pad_size > 0:
        x_active_mask = torch.nn.functional.pad(
            x_active_mask, (0, pad_size), value=False
        )
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
    # split hidden_states and x_active_mask if tp_size > 1
    if tp_size > 1:
        split_hidden_states = torch.tensor_split(hidden_states, tp_size, dim=0)
        split_x_active_mask = torch.tensor_split(x_active_mask, tp_size, dim=0)
        hidden_states = split_hidden_states[tp_rank]
        x_active_mask = split_x_active_mask[tp_rank]
    return hidden_states, split_hidden_states, num_tokens, x_active_mask, moe_group_name


def moe_finalize(
    split_hidden_states: list,
    moe_output: torch.Tensor,
    num_tokens: int,
    ep_size: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
):
    if ep_size > 1:
        if tp_size > 1:
            dist.all_gather(list(split_hidden_states), moe_output, tp_group)
            moe_output = torch.cat(split_hidden_states, dim=0)
        moe_output = moe_output[:num_tokens, :]
    return moe_output


def fused_moe_tp(
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    num_experts = gate_up_weights.size(0)
    active_num = hidden_states.size(0) * topk
    # do renormalize
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
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
    mlp_output = apply_mlp(
        expanded_hidden_states,
        gate_up_weights,
        down_weights,
        expert_tokens,
        group_list_type,
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
):
    # do renormalize
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    # distribute dispatch
    num_local_experts = gate_up_weights.size(0)
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
    mlp_output = apply_mlp(
        expanded_hidden_states,
        gate_up_weights,
        down_weights,
        expert_tokens,
        group_list_type,
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
):
    num_local_experts = gate_up_weights.size(0)
    num_experts = num_local_experts * ep_size

    def dispatch(hidden_states, topk_ids):
        # dispatch pre-process
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=num_experts, min=0, max=num_experts
        )
        num_out_tokens = topk_ids.numel()
        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = torch.empty(
            (num_local_tokens_per_expert.size(0) * ep_size,),
            dtype=num_local_tokens_per_expert.dtype,
            device=torch.npu.current_device(),
        )
        torch.distributed.all_gather_into_tensor(
            num_global_tokens_per_expert, num_local_tokens_per_expert, ep_group
        )
        num_global_tokens_per_expert = num_global_tokens_per_expert.reshape(
            ep_size, num_experts
        )
        hidden_shape_before_permute = hidden_states.shape
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :,
            num_local_experts * ep_rank : num_local_experts * (ep_rank + 1),
        ]
        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
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
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # dispatch post-process
        if num_local_experts <= 1:
            return global_input_tokens, None

        global_input_tokens, reversed_global_input_permutation_mapping = (
            torch.ops.npu.npu_moe_token_permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )
        )

        context_metadata = {
            "hidden_shape_before_permute": hidden_shape_before_permute,
            "input_splits": input_splits,
            "output_splits": output_splits,
            "reversed_local_input_permutation_mapping": reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping": reversed_global_input_permutation_mapping,
        }

        return {
            "hidden_states": global_input_tokens,
            "dynamic_scale": None,
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
        hidden_states.untyped_storage().resize_(0)

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
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        # distribute dispatch
        dispatched_outputs = dispatch(hidden_states, topk_ids)

        # MLP
        mlp_output = apply_mlp(
            dispatched_outputs["hidden_states"],
            gate_up_weights,
            down_weights,
            dispatched_outputs["group_list"].to(torch.int64),
            dispatched_outputs["group_list_type"],
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
