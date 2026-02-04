# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
import torch
from torch import Tensor

from typing import Any, Dict, List

from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin, next_power_of_2

BuffType = Dict[str, Tensor]


def get_camb_compatible_size(n: int):
    """Get camb compatible size (use power of 2 for simplicity)."""
    return next_power_of_2(n)


def CambCudaGraphMixin_make_buffers_cudagraph(
    self, graph_meta: CudaGraphMeta, *args, **kwargs
) -> BuffType:
    """make cudagraph buffers from forward inputs."""
    max_batches = graph_meta.max_batchs
    max_tokens = graph_meta.max_tokens
    num_blocks = graph_meta.num_blocks
    device = graph_meta.device

    input_buffers: BuffType = dict()
    input_buffers["input_ids"] = torch.zeros(
        1, max_tokens, dtype=torch.int32, device=device
    )
    input_buffers["position_ids"] = torch.ones(
        (1, max_tokens), dtype=torch.int32, device=device
    )

    input_buffers["block_offsets"] = torch.zeros(
        (max_batches, num_blocks), dtype=torch.int32, device=device
    )

    # in attn_metadata, we mock q_start_loc as cu_seqlens so need to add 1
    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["q_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["kv_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )
    # critical to set negative for kv_start_indices
    # if we don't set it, two batches with same input tokens
    # will result in different answer
    # NOTE: size should be max_batches, not max_tokens (fix for VL models)
    input_buffers["kv_start_indices"] = -torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    return input_buffers


def CambCudaGraphMixin_fill_buffers_cudagraph(
    self,
    graph_meta: CudaGraphMeta,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: List,
    attn_metadata: Any,
    inputs_embeds: Tensor,
    **kwargs
) -> Dict[str, Tensor]:
    """fill cudagraph buffers from forward inputs."""
    block_offsets: Tensor = attn_metadata.block_offsets
    q_start_loc: Tensor = attn_metadata.q_start_loc
    q_seqlens: Tensor = attn_metadata.q_seqlens
    kv_seqlens: Tensor = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    input_buffers["q_seqlens"][:batch_size] = q_seqlens
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens
    input_buffers["q_start_loc"][: batch_size + 1] = q_start_loc
    # Fix: use batch_size instead of num_tokens for kv_start_indices
    # kv_start_indices has shape (batch_size,), not (num_tokens,)
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds

    # below only used for capture graph
    # create inputs
    # Fix: compute new_batch_size based on batch_size, not num_tokens
    # This is critical for correct graph capture and replay
    new_batch_size = min(get_camb_compatible_size(batch_size), graph_meta.max_batchs)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.q_start_loc = input_buffers["q_start_loc"][: new_batch_size + 1]
    attn_metadata.q_seqlens = input_buffers["q_seqlens"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]

    # Fix: max_kv_seq_len is Python int, which gets fixed during graph capture.
    # According to torch_mlu_ops API:
    #   "If max_contxt_len <= 0, indicating an uncertain maximum value of context_lens."
    # So we set it to -1, and let the kernel compute max from kv_seqlens tensor internally.
    # This way, the kernel will always use the correct max value from the updated tensor.
    attn_metadata.max_kv_seq_len = -1
    attn_metadata.max_q_seq_len = -1  # Similarly for max_q_seq_len
    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
    )

    # For decoding, input_ids and position_ids have shape [1, num_tokens]
    # where num_tokens typically equals batch_size
    new_inputs["input_ids"] = input_buffers["input_ids"][:, :new_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :new_batch_size]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][:, :new_batch_size]

    new_inputs.update(kwargs)

    return new_inputs


def CambCudaGraphMixin_update_context_cudagraph(self, graph_meta, context):
    """update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    # Fix: also update block_offsets and kv_start_indices (reference ascend implementation)
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]


CudaGraphMixin.make_buffers_cudagraph = CambCudaGraphMixin_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph = CambCudaGraphMixin_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = CambCudaGraphMixin_update_context_cudagraph
