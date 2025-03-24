import torch

from . import pytorch_patch, camb_ops

# TODO. weitao: camb torch-mlu-ops-v1.2.0 per_token_smooth_quantize need smooth_vec
SMOOTH_VEC = torch.ones(8192, dtype=torch.float32, device="mlu")


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def update_smooth(length):
    global SMOOTH_VEC
    if length > SMOOTH_VEC.shape[0]:
        SMOOTH_VEC = torch.ones(
            next_power_of_2(length), dtype=torch.float32, device="mlu"
        )
    return SMOOTH_VEC
