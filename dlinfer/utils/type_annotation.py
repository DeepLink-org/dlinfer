# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
from torch import Tensor
from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Union, Any, Tuple, Callable, Dict


class MoeType(Enum):
    NATIVE = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    MC2 = auto()
    TP = auto()
    UNDEFINED = auto()


linear_w8a8_scale_type = torch.Tensor
dynamic_quant_scale_type = torch.Tensor
