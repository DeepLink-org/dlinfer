import torch
from dlinfer.graph.dicp.dynamo_bridge.operator import Operator

aten = torch.ops.aten


def negative_in_shape(shape):
    for elem in shape:
        if elem < 0:
            return True
    return False


class Linear(Operator):
    def __init__(self):
        super().__init__("Linear")

    def infer_result(self, a, b, bias, trans_a, trans_b):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        out = torch.matmul(a, b)
        if bias:
            out = out + bias
        return out


class LinearAllReduce(Operator):
    def __init__(self):
        super().__init__("LinearAllReduce")

    def infer_result(self, x, weight, bias, group):
        out = torch.matmul(x, weight.t())
        if bias:
            out = out + bias
        return out


class AllReduce(Operator):
    def __init__(self):
        super().__init__("AllReduce")

    def infer_result(self, x, reduce_type, group):
        return x


class AclNnAdd(Operator):
    def __init__(self):
        super().__init__("AclNnAdd")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a + b


class AclNnSub(Operator):
    def __init__(self):
        super().__init__("AclNnSub")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a - b


class AclNnDiv(Operator):
    def __init__(self):
        super().__init__("AclNnDiv")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a / b


class AclNnMul(Operator):
    def __init__(self):
        super().__init__("AclNnMul")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a * b


class Add(Operator):
    def __init__(self):
        super().__init__("Add")

    def infer_result(self, a, b):
        return a + b


class AclNnAdds(Operator):
    def __init__(self):
        super().__init__("AclNnAdds")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a + b


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")

    def infer_result(self, a, b):
        return a - b


class AclNnSubs(Operator):
    def __init__(self):
        super().__init__("AclNnSubs")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a - b


class Div(Operator):
    def __init__(self):
        super().__init__("Div")

    def infer_result(self, a, b):
        return a / b


class AclNnDivs(Operator):
    def __init__(self):
        super().__init__("AclNnDivs")

    def infer_result(self, a, b):
        return a / b


class InplaceDiv(Operator):
    def __init__(self):
        super().__init__("InplaceDiv")

    def infer_result(self, a, b):
        return a / b


class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")

    def infer_result(self, a, b):
        return a * b


class Muls(Operator):
    def __init__(self):
        super().__init__("Muls")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a * b


class PowTensorScalar(Operator):
    def __init__(self):
        super().__init__("PowTensorScalar")

    def infer_result(self, a, b, dtype="FLOAT"):
        return torch.ops.aten.pow.Tensor_Scala(a, b)


class PowTensorTensor(Operator):
    def __init__(self):
        super().__init__("PowTensorTensor")

    def infer_result(self, a, b):
        return torch.ops.aten.pow.Tensor_Tensor(a, b)


class Max(Operator):
    def __init__(self):
        super().__init__("Max")

    def infer_result(self, x):
        return torch.ops.aten.max.default(x)


class Reciprocal(Operator):
    def __init__(self):
        super().__init__("Reciprocal")

    def infer_result(self, x):
        return torch.ops.aten.reciprocal.default(x)


class GtScalar(Operator):
    def __init__(self):
        super().__init__("GtScalar")

    def infer_result(self, x, y, dtype="FLOAT"):
        return torch.ops.aten.gt.Scalar(x, y)


class GeScalar(Operator):
    def __init__(self):
        super().__init__("GeScalar")

    def infer_result(self, x, y, dtype="FLOAT"):
        return torch.ops.aten.ge.Scalar(x, y)


class Where(Operator):
    def __init__(self):
        super().__init__("Where")

    def infer_result(self, cond, x, y):
        return torch.ops.aten.where.self(cond, x, y)


class Arange(Operator):
    def __init__(self):
        super().__init__("Arange")

    def infer_result(self, start, end, step, dtype):
        return torch.ops.aten.arange.start_step(start, end, step, dtype=dtype)


class Graph(Operator):
    def __init__(self):
        super().__init__("Graph")

    def infer_result(self, *args, **kwargs):
        if not isinstance(kwargs["output"], list):
            return kwargs["output"].meta["val"]
        else:
            res = [x.meta["val"] for x in kwargs["output"]]
            return tuple(res)


class Tuple(Operator):
    def __init__(self):
        super().__init__("Tuple")

    def infer_result(self, *args, **kwargs):
        res = [x.meta["val"] for x in args]
        return tuple(res)


class GetItem(Operator):
    def __init__(self):
        super().__init__("GetItem")

    def infer_result(self, x, index):
        return x[index]


class RmsNorm(Operator):
    def __init__(
        self,
    ):
        super().__init__("RmsNorm")

    def infer_result(self, x, weight, eps):
        return (x, x)


class AddRmsNorm(Operator):
    def __init__(
        self,
    ):
        super().__init__("AddRmsNorm")

    def infer_result(self, x, residual, gamma, eps):
        return (x, x)


class Rope(Operator):
    def __init__(
        self,
    ):
        super().__init__("Rope")

    def infer_result(self, query, key, cos, sin, seqlen):
        return (query, key)


class Inplace(Operator):
    def __init__(self):
        super().__init__("Inplace")

    def infer_result(self, input, target, input_index=-1, target_index=-1):
        if target_index == -1:
            return target
        return target[target_index]


class SelfAttentionPAEncoder(Operator):
    def __init__(self):
        super().__init__("SelfAttentionPAEncoder")

    def infer_result(
        self,
        query,
        key,
        value,
        seqlen,
        mask,
        q_head_num,
        kv_head_num,
        scale,
        head_size,
        head_size_v,
    ):
        return query.new_empty((query.shape[0], q_head_num, head_size_v))


class ReshapeAndCache(Operator):
    def __init__(self):
        super().__init__("ReshapeAndCache")

    def infer_result(self, key, value, key_cache, value_cache, kv_indices):
        return key_cache, value_cache


class MlaReshapeAndCache(Operator):
    def __init__(self):
        super().__init__("MlaReshapeAndCache")

    def infer_result(self, key, key_cache, kv_indices):
        return key_cache


class PagedAttention(Operator):
    def __init__(self):
        super().__init__("PagedAttention")

    def infer_result(
        self,
        query,
        key_cache,
        value_cache,
        block_table,
        context_len,
        mask,
        q_head_num,
        kv_head_num,
        scale,
        head_size,
        head_size_v,
    ):
        return query.new_empty((query.shape[0], q_head_num, head_size_v))


class Transpose(Operator):
    def __init__(self):
        super().__init__("Transpose")

    def infer_result(self, x, perm):
        return x.t()


class View(Operator):
    def __init__(self):
        super().__init__("View")

    def infer_result(self, x, size):
        return x.view(size)


class Unsqueeze(Operator):
    def __init__(self):
        super().__init__("Unsqueeze")

    def infer_result(self, x, dim, target_shape=None):
        return x.unsqueeze(dim)


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")

    def infer_result(self, x, dim, target_shape=None):
        return x.squeeze(dim)


class SplitSharing(Operator):
    def __init__(self):
        super().__init__("SplitSharing")

    def infer_result(self, x, size, dim):
        return x.split(size, dim=dim)


class SplitWithSize(Operator):
    def __init__(self):
        super().__init__("SplitWithSize")

    def infer_result(self, x, sizes, dim):
        return x.split_with_sizes(sizes, dim=dim)


class Swish(Operator):
    def __init__(self):
        super().__init__("Swish")

    def infer_result(self, x, scale=1.0, dim=-1):
        return x


class Swiglu(Operator):
    def __init__(self):
        super().__init__("Swiglu")

    def infer_result(self, x, dim):
        x_shape = x.shape
        x_shape[dim] = x_shape[dim] // 2
        return torch.empty(x_shape, device=x.device, dtype=x.dtype)


class Cast(Operator):
    def __init__(self):
        super().__init__("Cast")

    def infer_result(self, x, out_dtype):
        return x.to(out_dtype)


class Sin(Operator):
    def __init__(self):
        super().__init__("Sin")

    def infer_result(self, x):
        return x.sin()


class Cos(Operator):
    def __init__(self):
        super().__init__("Cos")

    def infer_result(self, x):
        return x.cos()


class Concat(Operator):
    def __init__(self):
        super().__init__("Concat")

    def infer_result(self, x, dim):
        return torch.cat(x, dim)


class BatchMatMul(Operator):
    def __init__(self):
        super().__init__("BatchMatMul")

    def infer_result(self, x1, x2):
        return x1 @ x2


class Gather(Operator):
    def __init__(self):
        super().__init__("Gather")

    def infer_result(self, x1, x2, axis):
        return torch.ops.aten.embedding.default(x1, x2, axis)


class Softmax(Operator):
    def __init__(self):
        super().__init__("Softmax")

    def infer_result(self, x, dim):
        return torch.softmax(x, dim=self.dim)


class Sort(Operator):
    def __init__(self):
        super().__init__("Sort")

    def infer_result(self, x, topk):
        value, index = torch.topk(x, topk)
        return value, index


class Slice(Operator):
    def __init__(self):
        super().__init__("Slice")

    def infer_result(self, x, dim, offsets, size):
        return torch.ops.aten.slice.Tensor(
            x, dim, offsets[dim], offsets[dim] + size[dim], 1
        )


class AclNnSlice(Operator):
    def __init__(self):
        super().__init__("AclNnSlice")

    def infer_result(self, x, dim, start, end, step):
        return torch.ops.aten.slice.Tensor(x, dim, start, end, step)


class IndexSelect(Operator):
    def __init__(self):
        super().__init__("IndexSelect")

    def infer_result(self, x, dim, index):
        indices = [None] * len(x.shape)
        indices[dim] = index
        return torch.ops.aten.index.Tensor(x, indices)


class Expand(Operator):
    def __init__(self):
        super().__init__("AclNnExpand")

    def infer_result(self, x, size):
        return x.expand(size)


class InplaceScatter(Operator):
    def __init__(self):
        super().__init__("InplaceScatter")

    def infer_result(self, x, dim, index, src):
        return x


class AclNnGather(Operator):
    def __init__(self):
        super().__init__("AclNnGather")

    def infer_result(self, x, dim, index):
        return index


class ScalarTensor(Operator):
    def __init__(self):
        super().__init__("ScalarTensor")

    def infer_result(self, x, dtype):
        return torch.empty(1, dtype=dtype, device="npu")


class ReduceSum(Operator):
    def __init__(self):
        super().__init__("ReduceSum")

    def infer_result(self, x, dim):
        return x.sum(dim)


class ReduceMax(Operator):
    def __init__(self):
        super().__init__("ReduceMax")

    def infer_result(self, x, dim):
        return x.amax(dim)


class ReduceMin(Operator):
    def __init__(self):
        super().__init__("ReduceMin")

    def infer_result(self, x, dim):
        return x.amin(dim)


class AclNnBincount(Operator):
    def __init__(self):
        super().__init__("AclNnBincount")

    def infer_result(self, x, weights, minlength):
        return torch.bincount(x, weights=weights, minlength=minlength)


class AclNnCumsum(Operator):
    def __init__(self):
        super().__init__("AclNnCumsum")

    def infer_result(self, x, dim, dtype):
        return torch.cumsum(x, dim, dtype=dtype)


class Zeros(Operator):
    def __init__(self):
        super().__init__("Zeros")

    def infer_result(self, size, dtype):
        return torch.ops.aten.zeros.default(size, dtype=dtype)


class ZerosLike(Operator):
    def __init__(self):
        super().__init__("ZerosLike")

    def infer_result(self, x):
        return x


class SliceScatter(Operator):
    def __init__(self):
        super().__init__("SliceScatter")

    def infer_result(self, x, data, dim, start, end, step):
        return torch.slice_scatter(x, data, dim=dim, start=start, end=end, step=step)


class AclNnInplaceIndexCopy(Operator):
    def __init__(self):
        super().__init__("AclNnInplaceIndexCopy")

    def infer_result(self, x, data, dim=0, start=None, end=None, step=1, index=None):
        return torch.slice_scatter(x, data, dim=dim, start=start, end=end, step=step)


class Renormalize(Operator):
    def __init__(self):
        super().__init__("Renormalize")

    def infer_result(self, x, dim):
        return x.sum(dim), x


class PrepareMoe(Operator):
    def __init__(self):
        super().__init__("PrepareMoe")

    def infer_result(self, x, num_experts):
        return (
            x.transpose(0, 1).to(torch.int32),
            x.to(torch.int32),
            torch.arange(0, num_experts, 1, dtype=torch.int32),
            torch.arange(0, num_experts, 1, dtype=torch.int32),
        )


class MoeInitRouting(Operator):
    def __init__(self):
        super().__init__("AclNnMoeInitRouting")

    def infer_result(self, x, row_ids, topk_ids, active_num, num_experts):
        return (
            x.repeat_interleave(topk_ids.size(1), dim=0),
            row_ids.flatten(),
            topk_ids.flatten(),
        )


class AclNnMoeTokenPermute(Operator):
    def __init__(self):
        super().__init__("AclNnMoeTokenPermute")

    def infer_result(self, x, topk_ids):
        return (
            x.repeat_interleave(topk_ids.size(1), dim=0),
            topk_ids.flatten().to(torch.int32),
        )


class AclNnGroupedMatmul(Operator):
    def __init__(self):
        super().__init__("AclNnGroupedMatmul")

    def infer_result(self, x, weights, group, split_item=2):
        return x.new_empty(x.size(0), weights.size(1))


class MoeFinalizeRouting(Operator):
    def __init__(self):
        super().__init__("AclNnMoeFinalizeRouting")

    def infer_result(
        self,
        down_proj,
        skip1,
        skip2,
        bias,
        topk_weights,
        expanded_row_idx,
        export_for_source_row,
    ):
        return skip1


class AclNnMoeTokenUnpermute(Operator):
    def __init__(self):
        super().__init__("AclNnMoeTokenUnpermute")

    def infer_result(
        self,
        permuted_tokens,
        sorted_indices,
        probs,
    ):
        tokens_num = probs.size(0)
        hidden_size = permuted_tokens.size(1)
        return permuted_tokens.new_empty((tokens_num, hidden_size))


class NewEmpty(Operator):
    def __init__(self):
        super().__init__("NewEmpty")

    def infer_result(self, x, size):
        return x.new_empty(size)


class EmptyLike(Operator):
    def __init__(self):
        super().__init__("NewEmpty")

    def infer_result(self, x, size):
        return x


class AclNnInplaceCopy(Operator):
    def __init__(self):
        super().__init__("AclNnInplaceCopy")

    def infer_result(self, dest, src):
        return dest
