# SPDX-License-Identifier: Apache-2.0
"""Clamped SwiGLU (GLM/DeepSeek-V4 style) fused Triton kernel for Ascend NPU.

Replaces the eager chain
    gate = gate.float().clamp_max(limit)
    up = up.float().clamp(min=-limit, max=limit)
    (gate * sigmoid(gate) * up).to(dtype)
with one kernel that keeps the exact arithmetic in fp32 registers and avoids
the per-call fp32 round trips.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_clamp_kernel(
    X,
    Y,
    N,
    LIMIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base = X + row * (2 * N)
    gate = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    y = gate * tl.sigmoid(gate) * up
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty), mask=mask)


def swiglu_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU on the last dim: x[..., :N] gate, x[..., N:] up."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1 or x2.stride(0) != shape[-1]:
        x2 = x2.contiguous()
    M, N2 = x2.shape
    assert N2 % 2 == 0
    N = N2 // 2
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, triton.cdiv(N, BLOCK_N))
    with torch.npu.device(x.device.index):
        _swiglu_clamp_kernel[grid](
            x2,
            out,
            N,
            LIMIT=limit,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
    return out.reshape(tuple(shape[:-1]) + (N,))
