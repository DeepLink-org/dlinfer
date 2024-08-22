import json, math
import torch
import torch_npu
from infer_ext.vendor.ascend import atb_infer_param
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence
from dataclasses import asdict


torch.classes.load_library('/data2/yaofengchen/workspaces/lmdeploy_InferExt/atb_models/output/atb_speed/lib/libatb_speed_torch.so')

class AtbSingleOperator:
    def __init__(self, op_name: str, op_type: str):
        self.op_name = op_name
        self.op_type = op_type
        self.param = {}
        self.input_names = []
        self.output_names = []
        self.has_host_inputs = False
        self.host_input_names = []
        self.has_reshape_inputs = False
        self.reshape_inputs = []
    
    def set_input(self, x):
        self.input_names = x 
    
    def set_output(self, x):
        self.output_names = x
        
    def add_input(self, x):
        self.input_names.append(x)
    
    def add_output(self, x):
        self.output_names.append(x)
    
    def set_param(self, x):
        if not isinstance(x, dict):
            x = atb_infer_param.to_dict(x)
        self.param = x

    def build(self):
        node = {
            "nodeType": "singleOperation",
            "value": {
                "name": self.op_name,
                "type": self.op_type,
                "param": self.param,
                "inputNames": self.input_names,
                "outputNames": self.output_names,
                "hasHostInputs": self.has_host_inputs,
                "hostInputNames": self.host_input_names,
                "hasReshapeInputs": self.has_reshape_inputs,
                "reshapeInputs": self.reshape_inputs,
            },
        }
        return node


def atb_rope_operation(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
):
    bs, head, dim = query_states.shape
    numKeyValueHeads = key_states.shape[1]
    rope_operation = torch.classes.OperationTorch.OperationTorch("RopeOperation")
    rope_params = json.dumps(asdict(atb_infer_param.RopeParam()))
    rope_operation.set_param(rope_params)
    query_states = query_states.view(bs, -1)
    key_states = key_states.view(bs, -1)
    seqlen = torch.tensor([bs], dtype= torch.int32, device=query_states.device)
    q_rope, k_rope = rope_operation.execute([query_states, key_states, cos, sin, seqlen])
    return q_rope.view(bs, head, dim), k_rope.view(bs, numKeyValueHeads, dim)


def atb_self_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support alibi_slopes yet")
    bs, _, query_dim = query.shape
    value_dim = value.shape[-1]
    query = query.view(bs, -1)
    key = key.view(bs, -1)
    value = value.view(bs, -1)
    self_attention_operation = torch.classes.OperationTorch.OperationTorch("SelfAttention")
    param = atb_infer_param.SelfAttentionParam()
    param.calcType = atb_infer_param.SelfAttentionCalcType.PA_ENCODER
    param.kernelType = atb_infer_param.SelfAttentionKernelType.KERNELTYPE_DEFAULT
    param.clampType = atb_infer_param.SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
    param.headNum = num_q_heads
    param.kvHeadNum = num_kv_heads
    param.qkScale = attn_qk_scale if attn_qk_scale else 1. / math.sqrt(query_dim)
    param.isTriuMask = 1

    if attn_mask is not None:
        param.maskType = atb_infer_param.SelfAttentionMaskType.MASK_TYPE_NORM
        self_attention_params = json.dumps(asdict(param))
        self_attention_operation.set_param(self_attention_params)
        mask = torch.stack(attn_mask).to(query.dtype)
        out = self_attention_operation.execute([query, key, value, mask, seq_len.to(torch.int32)])
    else:
        param.maskType = atb_infer_param.SelfAttentionMaskType.MASK_TYPE_UNDEFINED
        self_attention_params = json.dumps(asdict(param))
        self_attention_operation.set_param(self_attention_params)
        out = self_attention_operation.execute([query, key, value, seq_len.to(torch.int32)])
    attn_output.copy_(out[0].view(bs, num_q_heads, query_dim)[..., :value_dim])


def atb_paged_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not "
                           "support alibi_slopes yet")
    bs, _, query_dim = query.shape
    num_blocks = key_cache.shape[0] // block_size
    key_cache = key_cache.view(num_blocks, block_size, num_kv_heads, -1)
    value_cache_tmp = value_cache.view(num_blocks, block_size, num_kv_heads, -1)
    key_dim = key_cache.shape[-1]
    value_dim = value_cache_tmp.shape[-1]
    value_cache = torch.zeros_like(key_cache)
    value_cache[..., :value_dim] = value_cache_tmp[:]


    paged_attention_operation = torch.classes.OperationTorch.OperationTorch("PagedAttentionOperation")
    param = atb_infer_param.PagedAttentionParam()
    param.headNum = num_q_heads
    param.kvHeadNum = num_kv_heads
    param.qkScale = attn_qk_scale if attn_qk_scale else 1. / math.sqrt(query_dim)
    
    # if mask is not None:
    #     param.maskType = infer_param.PagedAttentionMaskType.MASK_TYPE_NORM
    #     op.set_input([query, key_cache, value_cache, block_table, context_len, mask])
    # else:
    param.maskType = atb_infer_param.PagedAttentionMaskType.UNDEFINED
    paged_attention_params = json.dumps(asdict(param))
    paged_attention_operation.set_param(paged_attention_params)
    block_table = block_table.to(torch.int32)
    context_len = kv_seq_len.to(torch.int32)
    # import pdb;pdb.set_trace()
    # print("============= befor paged_attention_operation ==============", flush=True)
    out = paged_attention_operation.execute([query, key_cache, value_cache, block_table, context_len])
    attn_output.copy_(out[0].view(bs, num_q_heads, query_dim)[..., :value_dim])
