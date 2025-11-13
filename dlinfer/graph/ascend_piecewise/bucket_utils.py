"""Ascend capture bucket sizing utilities (ported from vLLM-Ascend)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.utils import get_logger


_LOGGER = get_logger("dlinfer.ascend.bucket")

# Default limit from vLLM-Ascend: 2048 hardware streams minus a safety buffer.
DEFAULT_MAX_CAPTURE_GRAPHS = 1800
COMM_STREAM_BUFFER = 40


@dataclass
class DistSummary:
    dp: int = 1
    tp: int = 1
    ep: int = 1
    enable_expert_parallel: bool = False


def _sorted_unique(values: Iterable[int]) -> List[int]:
    return sorted({v for v in values if isinstance(v, int) and v > 0})


def _uniform_sample(values: Sequence[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    if target_len >= len(values):
        return list(values)
    if target_len == 1:
        return [values[-1]]

    step = (len(values) - 1) / (target_len - 1)
    indices = [round(i * step) for i in range(target_len)]
    indices[0] = 0
    indices[-1] = len(values) - 1

    sampled: List[int] = []
    for idx in indices:
        value = values[idx]
        if not sampled or sampled[-1] != value:
            sampled.append(value)
    return sampled


def _detect_moe(model_config: Optional[ModelConfig]) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False

    if getattr(hf_config, "num_experts", 0):
        return True

    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        lowered = " ".join(str(item) for item in architectures).lower()
        if "moe" in lowered or "expert" in lowered:
            return True

    try:
        to_dict = getattr(hf_config, "to_dict", None)
        if callable(to_dict):
            hf_dict = to_dict()
            return any("expert" in str(key).lower() for key in hf_dict.keys())
    except Exception:  # pragma: no cover - defensive
        _LOGGER.debug("MoE detection failed", exc_info=True)

    return False


def _get_num_layers(model_config: Optional[ModelConfig]) -> Optional[int]:
    candidates = [
        getattr(model_config, "num_layers", None),
        getattr(model_config, "num_hidden_layers", None),
    ]

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        candidates.append(getattr(hf_config, "num_hidden_layers", None))

    llm_config = getattr(model_config, "llm_config", None)
    if llm_config is not None:
        candidates.append(getattr(llm_config, "num_hidden_layers", None))

    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def _moe_multiplier(model_config: Optional[ModelConfig]) -> float:
    if model_config is None:
        return 1.0
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return 1.0

    experts_per_tok = getattr(hf_config, "num_experts_per_tok", None)
    if isinstance(experts_per_tok, int) and experts_per_tok > 0:
        return max(1.0, experts_per_tok / 4.0)
    return 1.0


def summarize_dist(dist_config) -> DistSummary:
    if dist_config is None:
        return DistSummary()

    dp = getattr(dist_config, "dp", 1) or 1
    tp = getattr(dist_config, "attn_tp", None) or getattr(dist_config, "tp", 1) or 1
    ep = getattr(dist_config, "ep", 1) or 1
    enable_expert_parallel = bool(getattr(dist_config, "enable_eplb", False) or ep > 1)
    return DistSummary(dp=dp, tp=tp, ep=ep, enable_expert_parallel=enable_expert_parallel)


def _compute_bucket_limit(
    original_sizes: Sequence[int],
    model_config: Optional[ModelConfig],
    dist_summary: DistSummary,
    max_capture_graphs: int,
) -> Tuple[List[int], dict]:
    if max_capture_graphs is None:
        max_capture_graphs = DEFAULT_MAX_CAPTURE_GRAPHS

    sizes = _sorted_unique(original_sizes)
    info = {
        "valid": bool(sizes),
        "reason": "",
        "max_capture_graphs": max_capture_graphs,
        "parallel_factor": None,
        "resources_per_graph": None,
        "num_comm_groups": None,
        "dp": dist_summary.dp,
        "tp": dist_summary.tp,
        "ep": dist_summary.ep,
    }

    if not sizes:
        info["reason"] = "empty_input"
        return [], info

    num_layers = _get_num_layers(model_config)
    if num_layers is None:
        info["reason"] = "unknown_num_layers"
        return list(sizes), info

    resources_per_graph = num_layers + 1
    is_moe = _detect_moe(model_config)
    if is_moe:
        resources_per_graph = int(resources_per_graph * _moe_multiplier(model_config))
        resources_per_graph = max(resources_per_graph, num_layers + 1)
    info["resources_per_graph"] = resources_per_graph

    num_comm_groups = sum(size > 1 for size in (dist_summary.dp, dist_summary.tp))
    info["num_comm_groups"] = num_comm_groups

    hccl_mode = os.getenv("HCCL_OP_EXPANSION_MODE", "").upper()
    is_moe = _detect_moe(model_config)

    if hccl_mode == "AIV":
        parallel_factor = 1 + num_comm_groups + int(dist_summary.enable_expert_parallel)
        if is_moe and dist_summary.dp > 1:
            parallel_factor += 1
        else:
            max_capture_graphs = max_capture_graphs - parallel_factor * resources_per_graph

        parallel_factor = max(parallel_factor, 1)
        info.update({"parallel_factor": parallel_factor, "max_capture_graphs": max_capture_graphs})

        if max_capture_graphs <= 0:
            info["reason"] = "insufficient_streams"
            return [sizes[-1]], info

        limit = max_capture_graphs // resources_per_graph // parallel_factor
    else:
        denom = 1 + num_comm_groups * 2
        effective = max_capture_graphs - num_comm_groups * COMM_STREAM_BUFFER
        info.update({"parallel_factor": denom, "max_capture_graphs": effective})

        if denom <= 0 or effective <= 0:
            info["reason"] = "invalid_stream_estimate"
            return [sizes[-1]], info

        limit = effective // resources_per_graph // denom

    limit = max(1, limit)
    if limit >= len(sizes):
        info["reason"] = "no_truncation"
        return list(sizes), info

    sampled = _uniform_sample(sizes, limit)
    if not sampled:
        sampled = [sizes[-1]]
    info["reason"] = "truncated"
    info["final_bucket_count"] = len(sampled)
    return sampled, info


def limit_capture_buckets(
    default_sizes: Iterable[int],
    model_config: Optional[ModelConfig],
    dist_config=None,
    *,
    max_capture_graphs: int = DEFAULT_MAX_CAPTURE_GRAPHS,
) -> List[int]:
    if max_capture_graphs is None:
        max_capture_graphs = DEFAULT_MAX_CAPTURE_GRAPHS
    sizes, info = _compute_bucket_limit(
        default_sizes,
        model_config,
        summarize_dist(dist_config),
        max_capture_graphs,
    )

    reason = info.get("reason")
    if reason == "truncated":
        _LOGGER.info(
            "Ascend capture buckets truncated %s → %s (layers=%s, dp=%s, tp=%s, ep=%s, streams=%s)",
            list(_sorted_unique(default_sizes)),
            sizes,
            info.get("resources_per_graph"),
            info.get("dp"),
            info.get("tp"),
            info.get("ep"),
            info.get("max_capture_graphs"),
        )
    elif reason == "no_truncation":
        _LOGGER.info("Ascend capture buckets remain unchanged (count=%d).", len(sizes))
    elif reason not in (None, ""):
        _LOGGER.debug("Bucket limiter status=%s (result=%s)", reason, sizes)
    return sizes


def adjust_capture_batch_sizes(
    original_sizes: Iterable[int],
    model_config: Optional[ModelConfig],
    cache_config: Optional[CacheConfig] = None,
    dist_config=None,
    logger_: Optional[object] = None,
) -> List[int]:
    local_logger = logger_ or _LOGGER
    
    # Check for custom capture sizes from environment variable
    env_sizes_str = os.getenv("DLINFER_ASCEND_GRAPH_CAPTURE_SIZES", "")
    if env_sizes_str:
        try:
            # Parse comma-separated values and convert to integers
            env_sizes = [int(x.strip()) for x in env_sizes_str.split(",") if x.strip()]
            if env_sizes:
                local_logger.info(
                    "Using custom ACL graph capture sizes from environment: %s",
                    env_sizes,
                )
                # Validate that custom sizes are within reasonable bounds
                validated_sizes = _sorted_unique(env_sizes)
                if validated_sizes:
                    # When custom sizes are provided, use them directly without trimming
                    if cache_config is not None:
                        local_logger.info(
                            "Using custom ACL graph batches: %s (max_batches=%s)",
                            validated_sizes,
                            getattr(cache_config, "max_batches", "?"),
                        )
                    return validated_sizes
        except (ValueError, AttributeError) as e:
            local_logger.warning(
                "Failed to parse DLINFER_ASCEND_GRAPH_CAPTURE_SIZES=%s: %s. Using default sizes.",
                env_sizes_str, e
            )
    
    limited = limit_capture_buckets(
        original_sizes,
        model_config,
        dist_config,
        max_capture_graphs=DEFAULT_MAX_CAPTURE_GRAPHS,
    )

    if cache_config is not None and limited != list(original_sizes):
        local_logger.info(
            "Adjusted ACL graph batches %s → %s (max_batches=%s)",
            list(original_sizes),
            limited,
            getattr(cache_config, "max_batches", "?"),
        )
    return limited


limit_capture_bucket_list = limit_capture_buckets

__all__ = [
    "adjust_capture_batch_sizes",
    "limit_capture_buckets",
    "limit_capture_bucket_list",
]

