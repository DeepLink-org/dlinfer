import torch
import torch_npu

origin_torch_compile = torch.compile
from torch_npu.contrib import transfer_to_npu
torch.compile = origin_torch_compile

__all__ = [
    "apply_vendor_pytorch_patch",
    "comm_str",
    "device_str"
]

def apply_vendor_pytorch_patch():
    torch.Tensor.cuda = torch.Tensor.numpy
    torch.Tensor.is_cuda = torch.Tensor.is_npu
    torch.cuda = torch_npu.npu

comm_str = "hccl"
device_str = "npu"
