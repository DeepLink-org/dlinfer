from collections.abc import Sequence
from typing import Optional, Tuple, Union, List
from dlinfer.graph.dicp.dynamo_bridge.utils import get_memory_format, get_cast_dtype

import torch
import math

"""parse and get val"""


def remove_nested_parentheses(data):
    # ([a, d],) --> [a, d]
    # [[[a],]] --> a
    # [[['a',[['b']]], 'd'],] --> [['a', 'b'], 'd']
    if isinstance(data, (List, Tuple)) and len(data) == 1:
        return remove_nested_parentheses(data[0])
    elif isinstance(data, Tuple):
        if len(data) == 1 and not isinstance(data[0], (List, Tuple)):
            return remove_nested_parentheses(data[0])
        return tuple(remove_nested_parentheses(item) for item in data)
    elif isinstance(data, List):
        if len(data) == 1 and not isinstance(data[0], (List, Tuple)):
            return remove_nested_parentheses(data[0])
        return [remove_nested_parentheses(item) for item in data]
    else:
        return data


# in conversion.py, some ops' ("cast") inputs are ascend_type like 'FLOAT',but infer needs torch type
def ascend_type_to_torch(ascend_type: str) -> torch.dtype:
    ascend_type_map = {
        "BOOL": torch.bool,
        "INT64": torch.int64,
        "FLOAT": torch.float32,
        "FLOAT16": torch.float16,
        "INT32": torch.int32,
        "COMPLEX64": torch.complex64,
    }

    assert (
        ascend_type in ascend_type_map
    ), "unknow ascend_dtype in ascend_type_to_torch!"

    return ascend_type_map[ascend_type]


def get_fake_tensor_meta_val(
    x, req_dim=True, req_dtype=True
) -> Tuple[torch.Tensor, Union[torch.Size, list], int, Union[torch.dtype, type, None]]:
    x_shape = x.size() if hasattr(x, "size") else [1]
    x_dim = len(x_shape)
    x_dtype = x.dtype if hasattr(x, "dtype") else None
    return x, x_shape, x_dim, x_dtype


def get_op_const_arg_kwarg(
    const_arg,
) -> Tuple[list, torch.dtype, Union[list, None]]:
    """
    input:
        - const_arg: Tuple (new_args,kwargs)
            - new_args: Tuple, identical to input-"new_args" of operator Const (has 2 or 3 params currently)
            - kwargs: dict, identical to input-"kwargs" of operator Const
    output:
        - arg0: list, input attr such as axes,shape
        - arg1: torch dtype , e.g. torch.int32
        - arg2: list(optional), shape of arg0
    """
    new_args = const_arg[0]
    len_args = len(new_args)
    assert len_args >= 2 and len_args <= 4
    arg0, dtype = new_args[0], new_args[1]
    shape = new_args[2] if len(new_args) >= 3 else None
    ascend_format = new_args[3] if len(new_args) == 4 else None
    return arg0, dtype, shape, ascend_format


"""analyze dtype,format"""


def analyze_memory_format(tensor: torch.Tensor, operation: str) -> torch.memory_format:
    original_format = tensor.memory_format

    if operation == "transpose":
        # TODO: transpose
        ...
    elif operation == "permute":
        # TODO: permute
        ...

    return tensor.memory_format if tensor.is_contiguous() else original_format


def parse_variable(x):
    if isinstance(x, torch._subclasses.fake_tensor.FakeTensor):
        x, x_shape, _, x_dtype = get_fake_tensor_meta_val(x)
    elif isinstance(x, Tuple):  # Const input
        x, x_dtype, x_shape, _ = get_op_const_arg_kwarg(x)
    elif isinstance(x, (int, float)):  # Scalar input
        x, x_dtype, x_shape = x, type(x), []
    else:
        assert False, "unsupported input type!"
    x_shape = [] if x_shape is None else x_shape
    x_dtype = torch.float32 if x_dtype is None else x_dtype
    return x, x_shape, x_dtype


"""calculate size,stride,storage_offset"""


def get_broadcast_res_two_shape(shape1, shape2) -> Optional[list]:
    len1 = len(shape1)
    len2 = len(shape2)
    max_len = max(len1, len2)
    result_shape = []
    for i in range(-1, -max_len - 1, -1):
        dim1 = shape1[i] if i >= -len1 else 1
        dim2 = shape2[i] if i >= -len2 else 1
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result_shape.insert(0, max(dim1, dim2))
        else:
            print(torch.randn(shape1).shape, " ", torch.randn(shape2).shape, end=" ")
            assert False, "input shapes must be broadcastable!"
    return result_shape


def reduce_ops_output_size(
    x_shape, x_dim, dim: Union[None, Sequence, int], keepdim=False
):
    if dim is None or isinstance(dim, Sequence) and len(dim) == 0:
        if keepdim is True:
            shape = [1] * x_dim
        else:
            shape = []  # sum(all) need a scalar as ouput (no shape no stride)
    else:
        dim = [dim] if not isinstance(dim, Sequence) else dim
        dim = [(d + x_dim) % x_dim for d in dim]
        if keepdim is True:
            shape = [1 if r in dim else ori_size for r, ori_size in enumerate(x_shape)]
        else:
            shape = [
                x_shape[r]
                for r in range(x_dim)
                if r not in dim and r - x_dim not in dim
            ]
    return shape


def cal_stride_offset(new_shape: list, offset: list, res: torch.Tensor):
    stride = list(res.stride())
    ori_shape = list(res.size())
    new_offset = 0
    for s, off in zip(stride, offset):
        new_offset += s * off
    stride = [k for k, i, j in zip(stride, ori_shape, new_shape) if i != j]
    return stride, new_offset


"""binary&unary operators"""


def common_binary_op_infer(x1, x2, spec_dtype=None, spec_format=None) -> torch.Tensor:
    x1, x1_shape, x1_dtype = parse_variable(x1)
    x2, x2_shape, x2_dtype = parse_variable(x2)

    out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)
    dtype = get_cast_dtype(x1_dtype, x2_dtype) if not spec_dtype else spec_dtype
    if spec_format:
        memory_format = spec_format
    else:
        memory_format = (
            get_memory_format(x1)
            if isinstance(x1, torch._subclasses.fake_tensor.FakeTensor)
            else torch.contiguous_format
        )
    return torch.empty(out_shape, dtype=dtype, memory_format=memory_format)


def common_unary_op_infer(
    x, spec_dtype=None, spec_format=None, spec_shape=None
) -> torch.Tensor:
    _, x_shape, _, x_dtype = get_fake_tensor_meta_val(x)
    return torch.empty(
        x_shape if not spec_shape else spec_shape,
        dtype=x_dtype if not spec_dtype else spec_dtype,
        memory_format=get_memory_format(x) if not spec_format else spec_format,
    )


def reduce_op_infer(x, dims, keepdim) -> torch.tensor:
    x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
    out_shape = reduce_ops_output_size(x_shape, x_dim, dims, keepdim)
    return torch.empty(out_shape, dtype=x_dtype, memory_format=get_memory_format(x))


"""other common utils"""


def close2(num, tar=0, rtol=0.00001):
    return math.fabs(num - tar) < rtol
