# SPDX-License-Identifier: Apache-2.0

from .chunk import chunk_gated_delta_rule
from .sigmoid_gating import fused_sigmoid_gating_delta_rule_update

__all__ = [
    "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
]
