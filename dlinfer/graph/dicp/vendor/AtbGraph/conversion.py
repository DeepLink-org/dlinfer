import re
import os
import functools
import operator
import _operator
import torch
import math
from typing import (
    Optional,
)
from torch.types import (
    Number,
)
import numpy as np
import sympy
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.immutable_collections import immutable_list
from dlinfer.graph.dicp.vendor.AtbGraph import atb_op
from dlinfer.graph.dicp.dynamo_bridge.utils import symint_in_shape, neg_in_shape, not_all_num_shape, process_sym_name
from dlinfer.graph.dicp.dynamo_bridge.utils import preprocess_expression, find_root_num, merge_disjoint_set
from dlinfer.graph.dicp.dynamo_bridge.conversion import register_conversion_impl
from dlinfer.graph.dicp.dynamo_bridge.op_transformer import SingleOpTransformer
from dlinfer.graph.dicp.vendor.AtbGraph import ext_ops

aten = torch.ops.aten
conversions = {}


def register_conversion(aten_fn_or_str):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        register_conversion_impl,
        conversions,
        aten_fn_or_str,
    )


def replace_sym_in_shape_if_only_one(sizes):
    unstable_size_flag  = [0 if (isinstance(size, int) and size != -1)
                           else 1 for size in sizes]
    sym_size_count = sum(unstable_size_flag)
    if sym_size_count == 1:
        # e.g. sizes = (19, s0, 32, 128) => (19, -1, 32, 128)
        target_index = unstable_size_flag.index(1)
        new_sizes = list(sizes)
        new_sizes[target_index] = -1
        return new_sizes
    return sizes


def replace_negative_one_when_fixed(origin_shape, new_shape):
    negative_one_count = sum([1 if (isinstance(size, int) and size == -1)
                              else 0 for size in new_shape])
    if negative_one_count == 0:
        return new_shape
    elif negative_one_count >= 2:
        raise RuntimeError("found more than two '-1' in shape")
    else:
        origin_shape_with_sym_int = \
            [size.node.meta['val'] if isinstance(size, torch.fx.Proxy)
             else size for size in origin_shape]
        new_shape_with_sym_int = \
            [size.node.meta['val'] if isinstance(size, torch.fx.Proxy)
             else size for size in new_shape]
        origin_shape_prod = functools.reduce(operator.mul, origin_shape_with_sym_int)
        new_shape_prod_without_negative_one = \
            functools.reduce(operator.mul, filter(lambda x: x != -1, new_shape_with_sym_int))
        negative_one_value = origin_shape_prod // new_shape_prod_without_negative_one
        if isinstance(negative_one_value, torch.SymInt):
            negative_one_value = negative_one_value.node.maybe_as_int()
        if negative_one_value is None:
            # negative one contains symint
            return new_shape
        else:
            return [negative_one_value if (isinstance(size, int) and size == -1)
                    else size for size in new_shape]


class AtenToAtbTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    @register_conversion(torch.ops.atb.linear.default)
    def linear(self, a, b, bias, trans_a, trans_b):
        return self.get_proxy(atb_op.Linear, (a, b, bias, trans_a, trans_b))

    @register_conversion(operator.getitem)
    def identity(self, x, idx):
        return self.get_proxy(atb_op.GetItem, (x, idx))

    @register_conversion("torch.ops.dlinfer.rms_norm.default")
    def npu_rms_norm(self, x, w, eps=1e-6):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (x, w, eps))
        return rms_norm

    @register_conversion("torch.ops.lmdeploy.apply_rotary_pos_emb.default")
    def apply_rotary_pos_emb(self, q, k, cos, sin, q_out, k_out):
        if (q_out is not None) or (k_out is not None):
            raise RuntimeError("apply_rotary_pos_emb doesn't support outplace version in graph mode")

        q_shape = list(q.node.meta['val'].shape)
        k_shape = list(k.node.meta['val'].shape)
        is_qk_require_reshape = len(q_shape) == 3
        if is_qk_require_reshape:
            assert isinstance(q_shape[1], int) and isinstance(q_shape[2], int)
        new_q = q if not is_qk_require_reshape else \
            self.get_proxy(atb_op.View, (q, (-1, q_shape[1] * q_shape[2])))
        new_k = k if not is_qk_require_reshape else \
            self.get_proxy(atb_op.View, (k, (-1, k_shape[1] * k_shape[2])))
        out = self.get_proxy(atb_op.Rope, (new_q, new_k, cos, sin, None))
        if is_qk_require_reshape:
            out_q = self.get_proxy(atb_op.GetItem, (out, 0))
            out_q = self.get_proxy(atb_op.View, (out_q, (-1, q_shape[1], q_shape[2])))
            out_k = self.get_proxy(atb_op.GetItem, (out, 1))
            out_k = self.get_proxy(atb_op.View, (out_k, (-1, k_shape[1], k_shape[2])))
            out = self.get_proxy(atb_op.Tuple, (out_q, out_k))
        return out

    @register_conversion("torch.ops.dlinfer.fill_kv_cache.default")
    def fill_kv_cache(self, key, value, key_cache, value_cache, kv_indices):
        key_cache_shape = key_cache.node.meta['val'].shape
        key_shape = key.node.meta['val'].shape
        key_cache_reshaped = self.get_proxy(atb_op.View, (key_cache,
            (key_cache_shape[0], key_cache_shape[1], key_shape[-2], key_shape[-1])))
        value_cache_shape = value_cache.node.meta['val'].shape
        value_shape = value.node.meta['val'].shape
        value_cache_reshaped = self.get_proxy(atb_op.View, (value_cache,
            (value_cache_shape[0], value_cache_shape[1], value_shape[-2], value_shape[-1])))
        out = self.get_proxy(atb_op.ReshapeAndCache,
            (key, value, key_cache_reshaped, value_cache_reshaped, kv_indices))
        return out

    @register_conversion("torch.ops.dlinfer.paged_decode_attention.default")
    def paged_attention_decode(self, query, key_cache, value_cache, block_table, block_size, kv_seq_len,
                               max_kv_seq_len, num_q_heads, num_kv_heads, softmax_scale, alibi_slopes, attn_output):
        q_head_num = num_q_heads
        kv_head_num = num_kv_heads
        scale = 1. / math.sqrt(query.node.meta['val'].shape[-1])
        k_shape = list(key_cache.node.meta['val'].shape)
        v_shape = list(value_cache.node.meta['val'].shape)
        is_kv_require_reshape = len(k_shape) == 3 or len(v_shape) == 3
        if is_kv_require_reshape:
            key_cache = self.get_proxy(atb_op.View, (key_cache, (k_shape[0], k_shape[1], kv_head_num, -1)))
            value_cache = self.get_proxy(atb_op.View, (value_cache, (v_shape[0], v_shape[1], kv_head_num, -1)))
        out = self.get_proxy(atb_op.PagedAttention, (query, key_cache, value_cache, block_table, kv_seq_len, None, q_head_num, kv_head_num, scale))
        return out

    @register_conversion(torch.ops.aten.t.default)
    def t(self, input):
        shape = fx_traceback.get_current_meta()['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.mm.default)
    def aten_mm(self, x, y):
        return self.get_proxy(atb_op.Linear, (x, y, None, False, False))

    @register_conversion(torch.ops.aten.add.Tensor)
    def aten_add_tensor(self, x, y):
        return self.get_proxy(atb_op.Add, (x, y))

    @register_conversion(torch.ops.aten.view.default)
    def aten_view(self, x, size):
        return self.get_proxy(atb_op.View, (x, size))

    @register_conversion(torch.ops.aten.split_with_sizes.default)
    def split_with_sizes(self, x, size, dim):
        assert len(size) == 2 or len(size) == 3
        assert len(set(size)) == 1
        split = self.get_proxy(atb_op.SplitSharing, (x, size, dim))
        return split

    @register_conversion("torch.ops.dlinfer.silu_and_mul.default")
    def silu_and_mul(self, gate_up, dim):
        split = self.get_proxy(atb_op.SplitSharing, (gate_up, [1, 1], dim))
        gate = self.get_proxy(atb_op.GetItem, (split, 0))
        up = self.get_proxy(atb_op.GetItem, (split, 1))
        act = self.get_proxy(atb_op.Swish, (gate,))
        mul = self.get_proxy(atb_op.Mul, (act, up))
        graph = self.get_proxy(atb_op.Graph, (split, gate, up, act, mul), {'output': mul})
        return mul

    @register_conversion("torch.ops.dlinfer.add_rms_norm.default")
    def dlinfer_add_rms_norm(self, x1, x2, gamma, epsilon):
        add = self.get_proxy(atb_op.Add, (x1, x2))
        norm = self.get_proxy(atb_op.RmsNorm, (add, gamma, epsilon))
        graph = self.get_proxy(atb_op.Graph, (add, norm), {'output': [norm, add], 'infer_shape': {"type": "equal", "value": [(0, 0), (0, 0)]}})
        return self.get_proxy(atb_op.Tuple, (norm, add))

    @register_conversion(torch.ops.aten._to_copy.default)
    def to_copy(self, x, dtype=None, layout=None, device=None):
        assert layout is None
        assert device is None
        if dtype is not None:
            return self.get_proxy(atb_op.Cast, (x, dtype))
        raise RuntimeError('not support yet!')

    @register_conversion(torch.ops.aten.sin.default)
    def sin(self, x):
        return self.get_proxy(atb_op.Sin, (x,))

    @register_conversion(torch.ops.aten.cos.default)
    def cos(self, x):
        return self.get_proxy(atb_op.Cos, (x,))

    @register_conversion(torch.ops.aten.cat.default)
    def cat(self, x, dim):
        return self.get_proxy(atb_op.Concat, (x, dim))

    @register_conversion(torch.ops.aten.bmm.default)
    def bmm(self, x1, x2):
        out = self.get_proxy(atb_op.BatchMatMul, (x1, x2))
        return out

    @register_conversion(torch.ops.aten.transpose.int)
    def transpose_int(self, input, dim_1, dim_2):
        shape = fx_traceback.get_current_meta()['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape[dim_1], permute_shape[dim_2] = permute_shape[dim_2], permute_shape[dim_1]
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.embedding.default)
    def embedding(self, weight, indices, axis):
        return self.get_proxy(atb_op.Gather, (weight, indices, axis))

    @register_conversion("torch.ops.lmdeploy.prefill_attention.default")
    def prefill_attention(self,
                          query,
                          key,
                          value,
                          attn_output,
                          k_cache,
                          v_cache,
                          block_offsets,
                          q_start_loc,
                          q_seq_len,
                          kv_seq_len,
                          max_q_seq_len,
                          block_size,
                          mask,
                          is_unpaged_prefill):
        # k_cache = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        # v_cache = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        # fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (key, value, k_cache, v_cache, kv_start_indices_1d))
        # inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, k_cache, 0))
        # inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, v_cache, 1))
        if is_unpaged_prefill:
            k_shape = key.node.meta['val'].shape
            num_q_heads = query.node.meta['val'].shape[-2]
            num_kv_heads = k_shape[-2]
            kv_head_size = k_shape[-1]

            query = self.get_proxy(atb_op.View, (query, [-1, num_q_heads * kv_head_size]))
            key = self.get_proxy(atb_op.View, (key, [-1, num_kv_heads * kv_head_size]))
            value = self.get_proxy(atb_op.View, (value, [-1, num_kv_heads * kv_head_size]))

            out = self.get_proxy(atb_op.SelfAttentionPAEncoder, (query, key, value, kv_seq_len, mask[0], num_q_heads, num_kv_heads))
        else:
            q_shape = list(query.node.meta['val'].shape)
            scale = 1. / math.sqrt(q_shape[-1])
            k_cache_shape = list(k_cache.node.meta['val'].shape)
            k_shape = list(key.node.meta['val'].shape)
            v_cache_shape = list(v_cache.node.meta['val'].shape)
            num_q_heads = q_shape[-2]
            num_kv_heads = k_shape[-2]

            is_kv_require_reshape = len(k_cache_shape) == 3 or len(v_cache_shape) == 3
            if is_kv_require_reshape:
                k_cache = self.get_proxy(atb_op.View, (k_cache, (k_cache_shape[0], k_cache_shape[1], num_kv_heads, -1)))
                v_cache = self.get_proxy(atb_op.View, (v_cache, (v_cache_shape[0], v_cache_shape[1], num_kv_heads, -1)))
            out = self.get_proxy(atb_op.PagedAttention, (query, k_cache, v_cache, block_offsets, kv_seq_len, mask[0], num_q_heads, num_kv_heads, scale))
        # graph = self.get_proxy(atb_op.Graph, (out,), {"output": [out]})
        return out

    @register_conversion(torch.ops.aten.unsqueeze.default)
    def unsqueeze(self, x, dim):
        return self.get_proxy(atb_op.Unsqueeze, (x, dim))

    @register_conversion(torch.ops.aten.squeeze.dim)
    def squeeze(self, x, dim):
        return self.get_proxy(atb_op.Squeeze, (x, dim))

    @register_conversion(torch.ops.aten.select.int)
    def select_int(self, x, dim, index):
        try:
            x_shape = x.node.meta['val'].shape
            first_dim = x_shape[0]
            if first_dim == 1 and dim == 0 and index == 0:
                # FIX(tangzhiyi):
                # Here, the "squeeze" operation should be used, but currently,
                # the AscendATB processing of InputReshape changes the original
                # tensor's descriptor. This leads to the squeeze operation being
                # called multiple times. A temporary solution is to use "view"
                # instead of "squeeze".
                # return self.get_proxy(atb_op.Squeeze, (x, 0))
                view_shape = [-1 if isinstance(x, torch.SymInt) else x for x in x_shape]
                del view_shape[0]
                return self.get_proxy(atb_op.View, (x, view_shape))
        except Exception as e:
            pass
        raise RuntimeError(f'torch.ops.aten.select.int not support {dim} {index} yet!')

    @register_conversion(torch.ops.aten.alias.default)
    def alias(self, x):
        # lowering through view
        shape = replace_sym_in_shape_if_only_one(x.node.meta['val'].shape)
        return self.get_proxy(atb_op.View, (x, shape))


class ViewSymIntTransformer(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.aten.view.default:
            args_0_shape = args[0].node.meta['val'].shape
            new_args_1 = replace_negative_one_when_fixed(args_0_shape, args[1])
            new_args_1 = replace_sym_in_shape_if_only_one(new_args_1)
            new_args = (args[0], new_args_1)
            return super().call_function(target, new_args, kwargs)
        return super().call_function(target, args, kwargs)
