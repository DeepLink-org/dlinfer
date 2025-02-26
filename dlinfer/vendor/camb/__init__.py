import torch

from . import pytorch_patch, camb_ops

# TODO. weitao: camb torch-mlu-ops-v1.2.0 per_token_smooth_quantize need smooth_vec
SMOOTH_VEC = torch.ones(8000, dtype=torch.float32, device="mlu")
