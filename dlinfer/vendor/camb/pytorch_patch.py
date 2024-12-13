import torch
import torch_mlu

origin_torch_compile = torch.compile
from torch_mlu.utils.gpu_migration import migration

torch.compile = origin_torch_compile
