# Copyright (c) 2024, DeepLink. All rights reserved.
from pathlib import Path

import torch
from . import pytorch_patch, torch_npu_ops, v4_ops  # v4_ops loads _C_ascend kernels
