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

    def infer_result(self, x, weight, bias):
        out = torch.matmul(x, weight.t())
        if bias:
            out = out + bias
        return out


class AllReduce(Operator):
    def __init__(self):
        super().__init__("AllReduce")

    def infer_result(self, x, reduce_type):
        return torch.ops._c10d_functional.all_reduce.default(x, reduce_type, "0")


class Add(Operator):
    def __init__(self):
        super().__init__("Add")

    def infer_result(self, a, b):
        return a + b


class Adds(Operator):
    def __init__(self):
        super().__init__("Adds")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a + b


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")

    def infer_result(self, a, b):
        return a - b


class Subs(Operator):
    def __init__(self):
        super().__init__("Subs")

    def infer_result(self, a, b, dtype="FLOAT"):
        return a - b


class Div(Operator):
    def __init__(self):
        super().__init__("Div")

    def infer_result(self, a, b):
        return a / b


class Divs(Operator):
    def __init__(self):
        super().__init__("Divs")

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


class Where(Operator):
    def __init__(self):
        super().__init__("Where")

    def infer_result(self, cond, x, y):
        return torch.ops.aten.where.self(cond, x, y)


class Arange(Operator):
    def __init__(self):
        super().__init__("Arange")

    def infer_result(self, start, end, step):
        return torch.ops.aten.arange.start_step(start, end, step, dtype=torch.int64)


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

    def infer_result(self, query, key, value, seqlen, mask, q_head_num, kv_head_num):
        return query


class ReshapeAndCache(Operator):
    def __init__(self):
        super().__init__("ReshapeAndCache")

    def infer_result(self, key, value, key_cache, value_cache, kv_indices):
        return key_cache, value_cache


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
    ):
        return query


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

    def infer_result(self, x, dim):
        return x.unsqueeze(dim)


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")

    def infer_result(self, x, dim):
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
