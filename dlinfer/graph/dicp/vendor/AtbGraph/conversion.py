import os
import functools
import operator
import torch
import math

import torch.fx
from torch.fx.immutable_collections import immutable_dict
from collections import OrderedDict
import torch.fx.traceback as fx_traceback
from dlinfer.graph.dicp.vendor.AtbGraph import atb_op

from dlinfer.graph.dicp.dynamo_bridge.conversion import register_conversion_impl
from dlinfer.graph.dicp.dynamo_bridge.op_transformer import SingleOpTransformer
from dlinfer.graph.dicp.vendor.AtbGraph import ext_ops
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.utils import (
    get_ascend_dtype,
    get_reduce_dim,
)


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
    unstable_size_flag = [
        0 if (isinstance(size, int) and size != -1) else 1 for size in sizes
    ]
    sym_size_count = sum(unstable_size_flag)
    if sym_size_count == 1:
        # e.g. sizes = (19, s0, 32, 128) => (19, -1, 32, 128)
        target_index = unstable_size_flag.index(1)
        new_sizes = list(sizes)
        new_sizes[target_index] = -1
        return new_sizes
    return sizes


def replace_negative_one_when_fixed(origin_shape, new_shape):
    negative_one_count = sum(
        [1 if (isinstance(size, int) and size == -1) else 0 for size in new_shape]
    )
    if negative_one_count == 0:
        return new_shape
    elif negative_one_count >= 2:
        raise RuntimeError("found more than two '-1' in shape")
    else:
        origin_shape_with_sym_int = [
            size.node.meta["val"] if isinstance(size, torch.fx.Proxy) else size
            for size in origin_shape
        ]
        new_shape_with_sym_int = [
            size.node.meta["val"] if isinstance(size, torch.fx.Proxy) else size
            for size in new_shape
        ]
        origin_shape_prod = functools.reduce(operator.mul, origin_shape_with_sym_int)
        new_shape_prod_without_negative_one = functools.reduce(
            operator.mul, filter(lambda x: x != -1, new_shape_with_sym_int)
        )
        negative_one_value = origin_shape_prod // new_shape_prod_without_negative_one
        if isinstance(negative_one_value, torch.SymInt):
            negative_one_value = negative_one_value.node.maybe_as_int()
        if negative_one_value is None:
            # negative one contains symint
            return new_shape
        else:
            return [
                negative_one_value if (isinstance(size, int) and size == -1) else size
                for size in new_shape
            ]


class AtenToAtbTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)
        self._register_binary_ops()
        self.use_torch_npu_launcher = (
            os.getenv("DICP_USE_TORCH_NPU_LAUNCHER", "1") == "1"
        )
        self.graph_op_group = None

    def get_proxy(self, target, args, kwargs=immutable_dict()):
        proxy = super().get_proxy(target, args, kwargs)
        if self.use_torch_npu_launcher:
            if target == atb_op.Graph:
                return proxy
            if isinstance(self.graph_op_group, OrderedDict):
                assert id(proxy) not in self.graph_op_group
                self.graph_op_group[id(proxy)] = proxy
        return proxy

    @register_conversion(torch.ops.atb.linear.default)
    def linear(self, a, b, bias, trans_a, trans_b):
        return self.get_proxy(atb_op.Linear, (a, b, bias, trans_a, trans_b))

    @register_conversion(torch.ops.atb.allreduce.default)
    def allreduce(self, x, reduce_type, group):
        return self.get_proxy(atb_op.AllReduce, (x, reduce_type, group))

    @register_conversion(operator.getitem)
    def identity(self, x, idx):
        return self.get_proxy(atb_op.GetItem, (x, idx))

    @register_conversion("torch.ops.dlinfer.rms_norm.default")
    def npu_rms_norm(self, x, w, eps=1e-6):
        self.graph_op_group = OrderedDict()
        rms_norm = self.get_proxy(atb_op.RmsNorm, (x, w, eps))
        return rms_norm

    @register_conversion("torch.ops.lmdeploy.apply_rotary_pos_emb.default")
    def apply_rotary_pos_emb(self, q, k, cos, sin, q_out, k_out):
        q_shape = list(q.node.meta["val"].shape)
        k_shape = list(k.node.meta["val"].shape)
        is_qk_require_reshape = len(q_shape) == 3
        if is_qk_require_reshape:
            assert isinstance(q_shape[1], int) and isinstance(q_shape[2], int)
        new_q = (
            q
            if not is_qk_require_reshape
            else self.get_proxy(atb_op.View, (q, (-1, q_shape[1] * q_shape[2])))
        )
        new_k = (
            k
            if not is_qk_require_reshape
            else self.get_proxy(atb_op.View, (k, (-1, k_shape[1] * k_shape[2])))
        )
        out = self.get_proxy(atb_op.Rope, (new_q, new_k, cos, sin, None))
        if is_qk_require_reshape:
            out_q = self.get_proxy(atb_op.GetItem, (out, 0))
            out_q = self.get_proxy(atb_op.View, (out_q, (-1, q_shape[1], q_shape[2])))
            out_k = self.get_proxy(atb_op.GetItem, (out, 1))
            out_k = self.get_proxy(atb_op.View, (out_k, (-1, k_shape[1], k_shape[2])))
            out = self.get_proxy(atb_op.Tuple, (out_q, out_k))
        if (q_out is not None) and (k_out is not None):
            self.get_proxy(
                atb_op.AclNnInplaceCopy,
                (q_out, self.get_proxy(atb_op.GetItem, (out, 0))),
            )
            self.get_proxy(
                atb_op.AclNnInplaceCopy,
                (k_out, self.get_proxy(atb_op.GetItem, (out, 1))),
            )
            out = self.get_proxy(atb_op.Tuple, (q_out, k_out))
        return out

    @register_conversion("torch.ops.atb.inplace_div.default")
    def inplace_div(self, x, other):
        return self.get_proxy(atb_op.InplaceDiv, (x, other))

    @register_conversion("torch.ops.dlinfer.fill_kv_cache.default")
    def fill_kv_cache(
        self,
        key,
        value,
        key_cache,
        value_cache,
        kv_indices,
        k_scales_zeros,
        v_scales_zeros,
        quant_bits,
    ):
        key_cache_shape = key_cache.node.meta["val"].shape
        key_shape = key.node.meta["val"].shape
        key_cache = self.get_proxy(
            atb_op.View,
            (
                key_cache,
                (key_cache_shape[0], key_cache_shape[1], key_shape[-2], key_shape[-1]),
            ),
        )
        value_cache_shape = value_cache.node.meta["val"].shape
        value_shape = value.node.meta["val"].shape
        is_mla = key_shape[-1] != value_shape[-1]
        if not is_mla:
            value_cache = self.get_proxy(
                atb_op.View,
                (
                    value_cache,
                    (
                        value_cache_shape[0],
                        value_cache_shape[1],
                        value_shape[-2],
                        value_shape[-1],
                    ),
                ),
            )
            out = self.get_proxy(
                atb_op.ReshapeAndCache,
                (key, value, key_cache, value_cache, kv_indices),
            )
        else:
            out = self.get_proxy(
                atb_op.MlaReshapeAndCache,
                (key, key_cache, kv_indices),
            )
            out = self.get_proxy(atb_op.Tuple, (out, value_cache))
        return out

    @register_conversion("torch.ops.dlinfer.paged_decode_attention.default")
    def paged_attention_decode(
        self,
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale,
        alibi_slopes,
        attn_output,
        kv_scales,
        kv_zeros,
        quant_bits,
    ):
        scale = (
            1.0 / math.sqrt(query.node.meta["val"].shape[-1])
            if softmax_scale is None
            else softmax_scale
        )
        k_shape = list(key_cache.node.meta["val"].shape)
        v_shape = list(value_cache.node.meta["val"].shape)
        head_size = query.node.meta["val"].shape[-1]
        is_kv_require_reshape = len(k_shape) == 3 or len(v_shape) == 3
        if is_kv_require_reshape:
            head_size_v = v_shape[-1] // num_kv_heads
            key_cache = self.get_proxy(
                atb_op.View,
                (key_cache, (k_shape[0], k_shape[1], num_kv_heads, head_size)),
            )
            value_cache = self.get_proxy(
                atb_op.View,
                (value_cache, (v_shape[0], v_shape[1], num_kv_heads, head_size_v)),
            )
        else:
            head_size_v = v_shape[-1]
        out = self.get_proxy(
            atb_op.PagedAttention,
            (
                query,
                key_cache,
                value_cache,
                block_table,
                kv_seq_len,
                None,
                num_q_heads,
                num_kv_heads,
                scale,
                head_size,
                head_size_v,
            ),
        )
        return out

    @register_conversion(torch.ops.aten.t.default)
    def t(self, input):
        shape = fx_traceback.get_current_meta()["val"].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.mm.default)
    def aten_mm(self, x, y):
        return self.get_proxy(atb_op.Linear, (x, y, None, False, False))

    def _register_binary_ops(self):
        binary_ops = {
            (torch.ops.aten.add.Tensor, "add"): (
                atb_op.Add,
                atb_op.AclNnAdds,
                atb_op.AclNnAdd,
            ),
            (torch.ops.aten.sub.Tensor, "sub"): (
                atb_op.Sub,
                atb_op.AclNnSubs,
                atb_op.AclNnSub,
            ),
            (torch.ops.aten.mul.Tensor, "mul"): (
                atb_op.Mul,
                atb_op.Muls,
                atb_op.AclNnMul,
            ),
            (torch.ops.aten.div.Tensor, "div"): (
                atb_op.Div,
                atb_op.AclNnDivs,
                atb_op.AclNnDiv,
            ),
        }

        for (aten_op, op_name), (tensor_op, scalar_op, aclnn_op) in binary_ops.items():

            def make_handler(tensor_op, scalar_op, aclnn_op):
                def handler(self, x, y):
                    atb_supported_dtype = [torch.float16, torch.bfloat16]
                    out_dtype = fx_traceback.get_current_meta()["val"].dtype
                    if x.node.meta["val"].dtype != out_dtype:
                        x = self.get_proxy(atb_op.Cast, (x, out_dtype))
                    if isinstance(y, torch.fx.Proxy):
                        if y.node.meta["val"].dtype != out_dtype:
                            y = self.get_proxy(atb_op.Cast, (y, out_dtype))
                        if out_dtype in atb_supported_dtype:
                            return self.get_proxy(tensor_op, (x, y))
                        else:
                            dtype = get_ascend_dtype(out_dtype)
                            return self.get_proxy(aclnn_op, (x, y, dtype))
                    else:
                        dtype = get_ascend_dtype(out_dtype)
                        return self.get_proxy(scalar_op, (x, y, dtype))

                return handler

            register_conversion(aten_op)(make_handler(tensor_op, scalar_op, aclnn_op))

    @register_conversion(torch.ops.aten.pow.Tensor_Scalar)
    def aten_pow_tensor_scalar(self, x, y):
        dtype = get_ascend_dtype(x.node.meta["val"].dtype)
        return self.get_proxy(atb_op.PowTensorScalar, (x, y, dtype))

    @register_conversion(torch.ops.aten.pow.Tensor_Tensor)
    def aten_pow_tensor_tensor(self, x, y):
        return self.get_proxy(atb_op.PowTensorTensor, (x, y))

    @register_conversion(torch.ops.aten.gt.Scalar)
    def aten_gt_scalar(self, x, y):
        dtype = get_ascend_dtype(x.node.meta["val"].dtype)
        if len(x.node.meta["val"].shape) == 0:
            x = self.get_proxy(atb_op.View, (x, [1]))
        return self.get_proxy(atb_op.GtScalar, (x, y, dtype))

    @register_conversion(torch.ops.aten.ge.Scalar)
    def aten_ge_scalar(self, x, y):
        dtype = get_ascend_dtype(x.node.meta["val"].dtype)
        if len(x.node.meta["val"].shape) == 0:
            x = self.get_proxy(atb_op.View, (x, [1]))
        return self.get_proxy(atb_op.GeScalar, (x, y, dtype))

    @register_conversion(torch.ops.aten.max.default)
    def aten_max(self, x):
        if len(x.node.meta["val"].shape) == 0:
            x = self.get_proxy(atb_op.View, (x, [1]))
        return self.get_proxy(atb_op.Max, (x,))

    @register_conversion(torch.ops.aten.reciprocal.default)
    def aten_reciprocal(self, x):
        return self.get_proxy(atb_op.Reciprocal, (x,))

    @register_conversion(torch.ops.aten.where.self)
    def aten_where(self, cond, x, y):
        return self.get_proxy(atb_op.Where, (cond, x, y))

    @register_conversion(torch.ops.aten.arange.start_step)
    def aten_arange_start_step(self, start, end, step, dtype, device, index=0):
        assert dtype == torch.int64
        assert index == 0
        return self.get_proxy(atb_op.Arange, (start, end, step, dtype))

    @register_conversion(torch.ops.aten.view.default)
    def aten_view(self, x, size):
        return self.get_proxy(atb_op.View, (x, size))

    @register_conversion(torch.ops.aten._unsafe_view.default)
    def aten_unsafe_view(self, x, size):
        return self.get_proxy(atb_op.View, (x, size))

    @register_conversion(torch.ops.aten.split_with_sizes.default)
    def split_with_sizes(self, x, size, dim):
        if len(set(size)) == 1 and (len(size) == 2 or len(size) == 3):
            return self.get_proxy(atb_op.SplitSharing, (x, size, dim))
        return self.get_proxy(atb_op.SplitWithSize, (x, size, dim))

    @register_conversion(torch.ops.aten.split.Tensor)
    def split_tensor(self, x, size, dim):
        assert isinstance(size, int)
        rank = len(x.node.meta["val"].shape)
        dim = dim if dim > 0 else dim + rank
        split_dim_shape = x.node.meta["val"].shape[dim]
        sizes = []
        while split_dim_shape > 0:
            sizes.append(min(size, split_dim_shape))
            split_dim_shape -= size
        return self.get_proxy(atb_op.SplitWithSize, (x, sizes, dim))

    @register_conversion("torch.ops.dlinfer.silu_and_mul.default")
    def silu_and_mul(self, gate_up, dim):
        return self.get_proxy(atb_op.Swiglu, (gate_up, dim))

    @register_conversion("torch.ops.dlinfer.add_rms_norm.default")
    def dlinfer_add_rms_norm(self, x1, x2, gamma, epsilon):
        add = self.get_proxy(atb_op.Add, (x1, x2))
        if self.use_torch_npu_launcher and len(self.graph_op_group) > 0:
            op_tuple = tuple(self.graph_op_group.values())
            graph = self.get_proxy(atb_op.Graph, op_tuple, {"output": add})
        self.graph_op_group = OrderedDict()
        norm = self.get_proxy(atb_op.RmsNorm, (add, gamma, epsilon))
        # FIXME(tangzhiyi11): Temporarily disable graph op for MOE precision issues
        # graph = self.get_proxy(
        #     atb_op.Graph,
        #     (add, norm),
        #     {
        #         "output": [norm, add],
        #         "infer_shape": {"type": "equal", "value": [(0, 0), (0, 0)]},
        #     },
        # )
        return self.get_proxy(atb_op.Tuple, (norm, add))

    @register_conversion(torch.ops.aten._to_copy.default)
    def to_copy(self, x, dtype=None, layout=None, device=None):
        assert layout is None
        assert device is None
        if dtype is not None:
            return self.get_proxy(atb_op.Cast, (x, dtype))
        raise RuntimeError("not support yet!")

    @register_conversion("torch.ops.npu.npu_dtype_cast.default")
    def npu_dtype_cast(self, x, dtype=None, layout=None, device=None):
        assert layout is None
        assert device is None
        if dtype is not None:
            return self.get_proxy(atb_op.Cast, (x, dtype))
        raise RuntimeError("not support yet!")

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
        return self.get_proxy(atb_op.BatchMatMul, (x1, x2))

    @register_conversion(torch.ops.aten.transpose.int)
    def transpose_int(self, input, dim_1, dim_2):
        shape = fx_traceback.get_current_meta()["val"].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape[dim_1], permute_shape[dim_2] = (
            permute_shape[dim_2],
            permute_shape[dim_1],
        )
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.embedding.default)
    def embedding(self, weight, indices, padding_idx=None):
        # The padding_idx parameter is not supported now.
        return self.get_proxy(atb_op.Gather, (weight, indices, 0))

    @register_conversion("torch.ops.lmdeploy.prefill_attention.default")
    def prefill_attention(
        self,
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
        cu_seq_lens_kv,
        max_q_seq_len,
        max_kv_seq_len,
        block_size,
        mask,
        softmax_scale,
        is_unpaged_prefill,
        kv_scales,
        kv_zeros,
        quant_bits,
    ):
        mask = mask[0]
        scale = (
            softmax_scale
            if softmax_scale
            else 1.0 / math.sqrt(query.node.meta["val"].shape[-1])
        )
        _, num_q_heads, head_size = query.node.meta["val"].shape
        _, num_kv_heads, head_size_v = value.node.meta["val"].shape
        if query.node.meta["val"].dtype != mask.node.meta["val"].dtype:
            mask = self.get_proxy(atb_op.Cast, (mask, query.node.meta["val"].dtype))
        if is_unpaged_prefill:
            out = self.get_proxy(
                atb_op.SelfAttentionPAEncoder,
                (
                    query,
                    key,
                    value,
                    kv_seq_len,
                    mask,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    head_size,
                    head_size_v,
                ),
            )
        else:
            k_cache_shape = list(k_cache.node.meta["val"].shape)
            v_cache_shape = list(v_cache.node.meta["val"].shape)

            is_kv_require_reshape = len(k_cache_shape) == 3 or len(v_cache_shape) == 3
            if is_kv_require_reshape:
                k_cache = self.get_proxy(
                    atb_op.View,
                    (k_cache, (k_cache_shape[0], k_cache_shape[1], num_kv_heads, -1)),
                )
                v_cache = self.get_proxy(
                    atb_op.View,
                    (v_cache, (v_cache_shape[0], v_cache_shape[1], num_kv_heads, -1)),
                )
            out = self.get_proxy(
                atb_op.PagedAttention,
                (
                    query,
                    k_cache,
                    v_cache,
                    block_offsets,
                    kv_seq_len,
                    mask,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    head_size,
                    head_size_v,
                ),
            )
        # graph = self.get_proxy(atb_op.Graph, (out,), {"output": [out]})
        return out

    @register_conversion(torch.ops.aten.unsqueeze.default)
    def unsqueeze(self, x, dim):
        x_shape = list(x.node.meta["val"].shape)
        target_dim = dim if dim >= 0 else dim + len(x_shape) + 1
        x_shape.insert(target_dim, 1)
        x_shape = [str(x) for x in x_shape]
        return self.get_proxy(atb_op.Unsqueeze, (x, dim, x_shape))

    @register_conversion(torch.ops.aten.squeeze.dim)
    def squeeze(self, x, dim):
        x_shape = list(x.node.meta["val"].shape)
        target_dim = dim if dim >= 0 else dim + len(x_shape)
        x_shape.pop(target_dim)
        x_shape = [str(x) for x in x_shape]
        return self.get_proxy(atb_op.Squeeze, (x, dim, x_shape))

    @register_conversion(torch.ops.aten.select.int)
    def select_int(self, x, dim, index):
        try:
            x_shape = x.node.meta["val"].shape
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
        raise RuntimeError(f"torch.ops.aten.select.int not support {dim} {index} yet!")

    @register_conversion(torch.ops.aten.slice.Tensor)
    def slice_tensor(self, x, dim, start, end, step=1):
        dtype = fx_traceback.get_current_meta()["val"].dtype
        if dtype == torch.int64 or step != 1:
            return self.get_proxy(atb_op.AclNnSlice, (x, dim, start, end, step))
        x_shape = x.node.meta["val"].shape
        offsets = [0] * len(x_shape)
        size = [-1] * len(x_shape)

        offsets[dim] = start
        if end >= 9223372036854775807:
            size[dim] = -1
        else:
            size[dim] = end - start

        return self.get_proxy(atb_op.Slice, (x, dim, offsets, size))

    @register_conversion(torch.ops.aten.copy.default)
    def aten_copy(self, x, src):
        return src

    @register_conversion(torch.ops.aten.copy_.default)
    def aten_copy_(self, dest, src):
        return self.get_proxy(atb_op.AclNnInplaceCopy, (dest, src))

    @register_conversion(torch.ops.aten.clone.default)
    def aten_clone(self, x, memory_format=torch.contiguous_format):
        return x

    @register_conversion(torch.ops.aten.alias.default)
    def alias(self, x):
        # lowering through view
        shape = replace_sym_in_shape_if_only_one(x.node.meta["val"].shape)
        return self.get_proxy(atb_op.View, (x, shape))

    @register_conversion("torch.ops.dlinfer.linear.default")
    def dlinfer_linear(self, x, weight, bias, all_reduce, group):
        if all_reduce == False:
            return self.get_proxy(atb_op.Linear, (x, weight, bias, False, True))
        return self.get_proxy(atb_op.LinearAllReduce, (x, weight, bias, group))

    @register_conversion(torch.ops.aten.index.Tensor)
    def index_tensor(self, x, indices):
        dim = 0
        for index in indices:
            if index is None:
                continue
            x = self.get_proxy(atb_op.IndexSelect, (x, dim, index))
            dim += 1
        return x

    @register_conversion("torch.ops.dlinfer.fused_moe.default")
    def dlinfer_fused_moe(
        self,
        hidden_states,
        gate_up_weights,
        down_weights,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
    ):
        num_experts = gate_up_weights.node.meta["val"].shape[0]
        hidden_states_dtype = hidden_states.node.meta["val"].dtype
        if renormalize:
            reduce_dim = get_reduce_dim(topk_weights, -1)[0]
            topk_weights = self.get_proxy(
                atb_op.Renormalize, (topk_weights, reduce_dim)
            )
            topk_weights = self.get_proxy(atb_op.GetItem, (topk_weights, 1))

        topk_ids = self.get_proxy(atb_op.Cast, (topk_ids, torch.int32))
        pre_pare = self.get_proxy(atb_op.PrepareMoe, (topk_ids, num_experts))
        group = self.get_proxy(atb_op.GetItem, (pre_pare, 3))

        # moe token permute
        moe_init = self.get_proxy(
            atb_op.AclNnMoeTokenPermute, (hidden_states, topk_ids)
        )
        expanded_hidden_states = self.get_proxy(atb_op.GetItem, (moe_init, 0))
        expanded_row_idx = self.get_proxy(atb_op.GetItem, (moe_init, 1))

        # up sample
        gate_up_weights = self.get_proxy(atb_op.Transpose, (gate_up_weights, (0, 2, 1)))
        up_sample = self.get_proxy(
            atb_op.AclNnGroupedMatmul,
            (expanded_hidden_states, gate_up_weights, group, 2),
        )
        up_proj = self.get_proxy(atb_op.GetItem, (up_sample, 0))

        # activation
        gate_cache = self.silu_and_mul(up_proj, -1)

        # down sample
        down_weights = self.get_proxy(atb_op.Transpose, (down_weights, (0, 2, 1)))
        down_sample = self.get_proxy(
            atb_op.AclNnGroupedMatmul,
            (gate_cache, down_weights, group, 2),
        )
        down_proj = self.get_proxy(atb_op.GetItem, (down_sample, 0))

        # moe token unpermute
        topk_weights = self.get_proxy(atb_op.Cast, (topk_weights, hidden_states_dtype))
        moe_out = self.get_proxy(
            atb_op.AclNnMoeTokenUnpermute,
            (
                down_proj,
                expanded_row_idx,
                topk_weights,
            ),
        )
        return moe_out

    @register_conversion("torch.ops.dlinfer.moe_gating_topk_softmax.default")
    def dlinfer_moe_gating_topk_softmax(self, router_logits, top_k):
        routing_weights = self.get_proxy(atb_op.Softmax, (router_logits, -1))
        return self.get_proxy(atb_op.Sort, (routing_weights, top_k))

    @register_conversion(torch.ops.aten.expand.default)
    def aten_expand_default(self, x, size):
        x_shape = x.node.meta["val"].shape
        size = [
            x_shape[i] if size[i] == -1 and isinstance(x_shape[i], int) else size[i]
            for i in range(len(size))
        ]
        return self.get_proxy(atb_op.Expand, (x, size))

    @register_conversion(torch.ops.aten.gather.default)
    def aten_gather_default(self, x, dim, indices):
        return self.get_proxy(atb_op.AclNnGather, (x, dim, indices))

    @register_conversion("torch.ops.atb.inplace_scatter.default")
    def inplace_scatter(self, x, dim, index, src):
        return self.get_proxy(atb_op.InplaceScatter, (x, dim, index, src))

    @register_conversion(torch.ops.aten.scalar_tensor.default)
    def aten_scalar_tensor(self, x, dtype, layout, device):
        return self.get_proxy(atb_op.ScalarTensor, (float(x), dtype))

    @register_conversion(torch.ops.aten.sum.dim_IntList)
    def aten_reduce_sum(self, x, dim):
        dim = get_reduce_dim(x, dim)
        return self.get_proxy(atb_op.ReduceSum, (x, dim))

    @register_conversion(torch.ops.aten.amax.default)
    def aten_reduce_max(self, x, dim):
        dim = get_reduce_dim(x, dim)
        return self.get_proxy(atb_op.ReduceMax, (x, dim))

    @register_conversion(torch.ops.aten.amin.default)
    def aten_reduce_min(self, x, dim):
        dim = get_reduce_dim(x, dim)
        return self.get_proxy(atb_op.ReduceMin, (x, dim))

    @register_conversion(torch.ops.aten.bincount)
    def aten_bincount(self, x, weights=None, minlength=0):
        return self.get_proxy(atb_op.AclNnBincount, (x, weights, minlength))

    @register_conversion(torch.ops.aten.cumsum)
    def aten_cumsum(self, x, dim, dtype=None, out=None):
        out_dtype = fx_traceback.get_current_meta()["val"].dtype
        return self.get_proxy(atb_op.AclNnCumsum, (x, dim, out_dtype))

    @register_conversion(torch.ops.aten.zeros.default)
    def aten_zeros_default(self, size, dtype, device=None, pin_memory=False):
        return self.get_proxy(atb_op.Zeros, (size, dtype))

    @register_conversion(torch.ops.aten.zeros_like.default)
    def aten_zeros_like_default(self, x, pin_memory=False):
        return self.get_proxy(atb_op.ZerosLike, (x,))

    @register_conversion(torch.ops.aten.new_empty.default)
    def aten_new_empty(self, x, size, pin_memory=False):
        return self.get_proxy(atb_op.NewEmpty, (x, size))

    @register_conversion(torch.ops.aten.slice_scatter.default)
    def aten_slice_scatter(self, x, data, dim=0, start=None, end=None, step=1):
        return self.get_proxy(atb_op.SliceScatter, (x, data, dim, start, end, step))


class ViewSymIntTransformer(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.aten.view.default:
            args_0_shape = args[0].node.meta["val"].shape
            new_args_1 = replace_negative_one_when_fixed(args_0_shape, args[1])
            new_args_1 = replace_sym_in_shape_if_only_one(new_args_1)
            new_args = (args[0], new_args_1)
            return super().call_function(target, new_args, kwargs)
        return super().call_function(target, args, kwargs)
