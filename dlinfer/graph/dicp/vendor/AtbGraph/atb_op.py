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

class Add(Operator):
    def __init__(self):
        super().__init__("Add")
    
    def infer_result(self, a, b):
        return a + b

class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")
    
    def infer_result(self, a, b):
        return a * b

class Graph(Operator):
    def __init__(self):
        super().__init__("Graph")
        
    def infer_result(self, *args, **kwargs):
        if not isinstance(kwargs['output'], list):
            return kwargs['output'].meta['val']
        else:
            res = [x.meta['val'] for x in kwargs['output']]
            return tuple(res)

class Tuple(Operator):
    def __init__(self):
        super().__init__("Tuple")
    
    def infer_result(self, *args, **kwargs):
        res = [x.meta['val'] for x in args]
        return tuple(res)


class GetItem(Operator):
    def __init__(self):
        super().__init__("GetItem")

    def infer_result(self, x, index):
        return x[index]


class RmsNorm(Operator):
    def __init__(self,):
        super().__init__("RmsNorm")
    
    def infer_result(self, x, weight, eps):
        return (x, x)


class Rope(Operator):
    def __init__(self,):
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
    
    def infer_result(self, query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale):
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
