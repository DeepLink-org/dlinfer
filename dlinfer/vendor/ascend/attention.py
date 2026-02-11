import math
import torch
from dlinfer.utils.type_annotation import Tensor, Optional
from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import (
    AscendGraphRunner,
    get_graph_params,
    aclgraph_use_torch_npu_update,
)


def decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    scale_value: float,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    softmax_scale: float,
    attn_output: Tensor,
):
    if AscendGraphRunner.capturing and not aclgraph_use_torch_npu_update():
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
    elif AscendGraphRunner.capturing:
        bs, _, dim = query.shape
        block_num = key_cache.size(0)
        query = query.contiguous()
        attn_output = attn_output.contiguous()
        query = query.view(bs, 1, num_q_heads * dim)
        key_cache = key_cache.view(block_num, block_size, -1)
        value_cache = value_cache.view(block_num, block_size, -1)
        scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=None,
            block_table=block_table,
            input_layout="BSH",
            block_size=block_size,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=kv_seq_len,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=scale_value,
            sparse_mode=0,
        )
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
