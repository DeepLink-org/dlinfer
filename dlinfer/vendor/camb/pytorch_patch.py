import torch
import torch_mlu

origin_torch_compile = torch.compile
from torch_mlu.utils.model_transfer import transfer

torch.compile = origin_torch_compile
