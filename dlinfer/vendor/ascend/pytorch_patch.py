# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
import torch_npu

origin_torch_compile = torch.compile
from torch_npu.contrib import transfer_to_npu

torch.compile = origin_torch_compile
