import torch
import torch_mlu

origin_torch_compile = torch.compile
from torch_mlu.utils.model_transfer import transfer

torch.compile = origin_torch_compile

__all__ = ["apply_vendor_pytorch_patch", "comm_str", "device_str"]

def apply_vendor_pytorch_patch():
    torch.Tensor.cuda = torch.Tensor.numpy
    torch.Tensor.is_cuda = torch.Tensor.is_mlu
    torch.cuda = torch_mlu.mlu

comm_str = "cncl"
device_str = "mlu"

