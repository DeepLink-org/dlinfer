import torch

from .device_proxy import GetDeviceProxy, GetDeviceStaticProxy
from .distributed import apply_dist_patch


__all__ = [
    "apply_tensor_method_patch",
    "apply_torch_function_patch",
    "apply_dist_patch",
]


# mock device functions in generated/python_variable_methods.cpp
def apply_tensor_method_patch():
    torch.Tensor.to = GetDeviceProxy(torch.Tensor.to)
    torch.Tensor.is_pinned = GetDeviceProxy(torch.Tensor.is_pinned)
    torch.Tensor.pin_memory = GetDeviceProxy(torch.Tensor.pin_memory)

    torch.Tensor.new_tensor = GetDeviceProxy(torch.Tensor.new_tensor, pos=-1)
    torch.Tensor.new_empty = GetDeviceProxy(torch.Tensor.new_empty, pos=-1)
    torch.Tensor.new_empty_strided = GetDeviceProxy(
        torch.Tensor.new_empty_strided, pos=-1
    )
    torch.Tensor.new_full = GetDeviceProxy(torch.Tensor.new_full, pos=-1)
    torch.Tensor.new_ones = GetDeviceProxy(torch.Tensor.new_ones, pos=-1)
    torch.Tensor.new_zeros = GetDeviceProxy(torch.Tensor.new_zeros, pos=-1)
    # --- add other device func
    # legacy api
    torch.Tensor.new = GetDeviceProxy(torch.Tensor.new, pos=-1)

    torch.Tensor.cuda = torch.Tensor.npu
    torch.Tensor.is_cuda = torch.Tensor.is_npu


# mock device functions in generated/python_torch_functionsEverything.cpp
def apply_torch_function_patch():
    torch._C._nn._parse_to = GetDeviceProxy(torch._C._nn._parse_to, caller="static")
    torch.ones = GetDeviceStaticProxy(torch.ones)
    torch.ones_like = GetDeviceStaticProxy(torch.ones_like)
    torch.zeros = GetDeviceStaticProxy(torch.zeros)
    torch.zeros_like = GetDeviceStaticProxy(torch.zeros_like)
    torch.as_tensor = GetDeviceStaticProxy(torch.as_tensor)
    torch.tensor = GetDeviceStaticProxy(torch.tensor)
    torch.arange = GetDeviceStaticProxy(torch.arange)
    torch.range = GetDeviceStaticProxy(torch.range)

    torch.empty = GetDeviceStaticProxy(torch.empty)
    torch.empty_like = GetDeviceStaticProxy(torch.empty_like)
    torch.empty_strided = GetDeviceStaticProxy(torch.empty_strided)

    torch.eye = GetDeviceStaticProxy(torch.eye)
    torch.full = GetDeviceStaticProxy(torch.full)
    torch.full_like = GetDeviceStaticProxy(torch.full_like)
    torch.from_file = GetDeviceStaticProxy(torch.from_file)
    torch._pin_memory = GetDeviceStaticProxy(torch._pin_memory)
    torch.scalar_tensor = GetDeviceStaticProxy(torch.scalar_tensor)

    torch.rand = GetDeviceStaticProxy(torch.rand)
    torch.rand_like = GetDeviceStaticProxy(torch.rand_like)
    torch.randint = GetDeviceStaticProxy(torch.randint)
    torch.randint_like = GetDeviceStaticProxy(torch.randint_like)
    torch.randn = GetDeviceStaticProxy(torch.randn)
    torch.randn_like = GetDeviceStaticProxy(torch.randn_like)
    torch.randperm = GetDeviceStaticProxy(torch.randperm)
