# Copyright (c) 2024, DeepLink. All rights reserved.
from pathlib import Path

import torch
from . import pytorch_patch, torch_npu_ops

torch.ops.load_library(str(Path(__file__).parent / "ascend_extension.so"))
