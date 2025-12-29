# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any, Tuple, Callable, Dict


@dataclass
class DlinferDistContext:
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1

    dp_rank: int = 0
    tp_rank: int = 0
    ep_rank: int = 0

    max_tokens_accros_dp: int = 1

    tp_group: torch.distributed.ProcessGroup = None
    ep_group: torch.distributed.ProcessGroup = None


linear_w8a8_scale_type = torch.Tensor
dynamic_quant_scale_type = torch.Tensor
