# Copyright (c) 2024, DeepLink. All rights reserved.
from packaging import version

import torch
import torch_npu

origin_torch_compile = torch.compile
from torch_npu.contrib import transfer_to_npu

torch.compile = origin_torch_compile

if version.parse(torch.__version__) >= version.parse("2.2.0"):
    from importlib import import_module

    target_module_str = "torch.utils._triton"
    target_module = import_module(target_module_str)
    func_str = "has_triton"

    def has_triton():
        return False

    setattr(target_module, func_str, has_triton)
