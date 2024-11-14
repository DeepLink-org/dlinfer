import math

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Sequence


torch._dynamo.config.suppress_errors = False


# atb mm
@torch._custom_op.impl.custom_op("atb::linear")
def linear(
    a: Tensor, b: Tensor, bias: Tensor, trans_a: bool, trans_b: bool
) -> Tensor: ...


@linear.impl_abstract()
def atb_linear_abstract(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    return torch.matmul(a, b)


@linear.impl(["cpu", "cuda"])
def atb_linear_impl(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    out = torch.matmul(a, b)
    if bias:
        out = out + bias
    return out


# atb allreduce
@torch._custom_op.impl.custom_op("atb::allreduce")
def allreduce(
    x: Tensor,
    reduce_type: str,
) -> Tensor: ...


@allreduce.impl_abstract()
def atb_allreduce_abstract(x, reduce_type):
    return torch.ops._c10d_functional.all_reduce.default(x, reduce_type, 0)


@allreduce.impl(["cpu", "cuda"])
def atb_allreduce_impl(x, reduce_type):
    all_reduce = torch.ops._c10d_functional.all_reduce.default(x, reduce_type, 0)
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
    return wait_tensor
