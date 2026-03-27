# SPDX-License-Identifier: Apache-2.0
# Triton operators for Qwen3.5 inference on Ascend NPU.
# See individual files for kernel sources and licenses.

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update_npu",
    "chunk_gated_delta_rule",
    "RMSNormGated",
]

from .fla.chunk import chunk_gated_delta_rule
from .rms_norm_gated import RMSNormGated
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
