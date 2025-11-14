"""Ascend capture bucket sizing utilities (ported from vLLM-Ascend)."""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Sequence

from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.utils import get_logger


_LOGGER = get_logger("dlinfer.ascend.bucket")

# Default limit from vLLM-Ascend: 2048 hardware streams minus a safety buffer.
DEFAULT_MAX_CAPTURE_GRAPHS = 1800
COMM_STREAM_BUFFER = 40
_SHARED_EXPERT_ENV = "DLINFER_ASCEND_MULTISTREAM_SHARED_EXPERT"


def _collect_positive(values: Iterable[int]) -> List[int]:
    return [value for value in values if isinstance(value, int) and value > 0]


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

    return [values[idx] for idx in indices]


def _contains_expert(config) -> bool:
    if isinstance(config, dict):
        for key, value in config.items():
            if "expert" in str(key).lower():
                return True
            if _contains_expert(value):
                return True
    return False


def _is_moe_model(model_config: Optional[ModelConfig]) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False
    to_dict = getattr(hf_config, "to_dict", None)
    if callable(to_dict):
        try:
            return _contains_expert(to_dict())
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


def _get_architecture_name(model_config: Optional[ModelConfig]) -> str:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        architectures = getattr(hf_config, "architectures", None)
        if isinstance(architectures, (list, tuple)) and architectures:
            return str(architectures[0])
    return "unknown"


def _shared_expert_overlap_enabled() -> bool:
    env = os.getenv(_SHARED_EXPERT_ENV)
    if not env:
        return False
    try:
        return bool(int(env))
    except ValueError:
        lowered = env.strip().lower()
        return lowered in {"true", "on", "yes"}


def _parallel_config(dist_config) -> tuple[int, int, bool]:
    dp = 1
    tp = 1
    enable_expert_parallel = False
    if dist_config is not None:
        dp = getattr(dist_config, "dp", 1) or 1
        tp = getattr(dist_config, "attn_tp", None) or getattr(dist_config, "tp", 1) or 1
        ep = getattr(dist_config, "ep", 1) or 1
        enable_expert_parallel = bool(
            getattr(dist_config, "enable_expert_parallel", False)
            or getattr(dist_config, "enable_eplb", False)
            or ep > 1
        )
    return dp, tp, enable_expert_parallel


def limit_capture_buckets(
    default_sizes: Iterable[int],
    model_config: Optional[ModelConfig],
    dist_config=None,
    *,
    max_capture_graphs: int = DEFAULT_MAX_CAPTURE_GRAPHS,
) -> List[int]:
    if max_capture_graphs is None:
        max_capture_graphs = DEFAULT_MAX_CAPTURE_GRAPHS

    sizes = _collect_positive(default_sizes)
    if not sizes:
        return []

    original_sizes = list(sizes)
    num_layers = _get_num_layers(model_config)
    if num_layers is None:
        _LOGGER.debug(
            "Unable to determine number of hidden layers; returning original capture sizes."
        )
        return list(sizes)

    dp, tp, enable_expert_parallel = _parallel_config(dist_config)
    num_comm_groups = sum(size > 1 for size in (dp, tp))
    resources_per_graph = num_layers + 1
    hccl_mode = os.getenv("HCCL_OP_EXPANSION_MODE", "").upper()
    is_moe = _is_moe_model(model_config)
    shared_overlap = _shared_expert_overlap_enabled()

    capture_limit = max_capture_graphs
    calc_info = {
        "model": _get_architecture_name(model_config),
        "layers": num_layers,
        "dp": dp,
        "tp": tp,
        "expert_parallel": enable_expert_parallel,
        "num_comm_groups": num_comm_groups,
        "is_moe": is_moe,
        "hccl_mode": hccl_mode or "DEFAULT",
        "shared_expert_overlap": shared_overlap,
        "resources_per_graph": resources_per_graph,
        "max_capture_graphs_input": max_capture_graphs,
    }

    if hccl_mode == "AIV":
        parallel_factor = 1 + num_comm_groups + int(enable_expert_parallel) + int(
            shared_overlap
        )
        # if is_moe and dp > 1:
        #     parallel_factor += 1
        # else:
        #     capture_limit -= parallel_factor * resources_per_graph
        # Tighter control to avoid graph capture failures caused by too many batches
        parallel_factor += 1
        capture_limit -= parallel_factor * resources_per_graph

        parallel_factor = max(parallel_factor, 1)
        max_num_batch_sizes = math.floor(
            capture_limit / resources_per_graph / parallel_factor
        )
        calc_info.update(
            {
                "parallel_factor": parallel_factor,
                "capture_limit_after_reserve": capture_limit,
            }
        )
        _LOGGER.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes,
        )
    else:
        effective = capture_limit - num_comm_groups * COMM_STREAM_BUFFER
        denom = 1 + num_comm_groups * 2
        max_num_batch_sizes = math.floor(
            effective / resources_per_graph / denom
        )
        calc_info.update(
            {
                "effective_stream_budget": effective,
                "denominator": denom,
            }
        )
        _LOGGER.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes,
        )
        _LOGGER.warning(
            "Currently, communication is performed using FFTS+ method, which reduces "
            "the number of available streams and, as a result, limits the range of runtime "
            "shapes that can be handled. To both improve communication performance and "
            "increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV."
        )

    if max_num_batch_sizes < 1:
        max_num_batch_sizes = 1

    truncated = max_num_batch_sizes < len(sizes)
    if not truncated:
        result = list(sizes)
        _LOGGER.info(
            "No adjustment needed for ACL graph batch sizes: layers=%d, count=%d",
            num_layers,
            len(sizes),
        )
    else:
        sampled = _uniform_sample(sizes, max_num_batch_sizes)
        if not sampled:
            sampled = [sizes[-1]]
        result = sampled
        _LOGGER.info(
            "Adjusted ACL graph batch sizes for %s model (layers: %d): %d → %d sizes",
            _get_architecture_name(model_config),
            num_layers,
            len(sizes),
            len(result),
        )

    calc_info.update(
        {
            "max_num_batch_sizes": max_num_batch_sizes,
            "sizes_before": original_sizes,
            "sizes_after": list(result),
            "final_bucket_count": len(result),
            "result": "truncated" if truncated else "unchanged",
        }
    )
    _LOGGER.info("ACL graph capture bucket summary: %s", calc_info)
    return result


def adjust_capture_batch_sizes(
    original_sizes: Iterable[int],
    model_config: Optional[ModelConfig],
    cache_config: Optional[CacheConfig] = None,
    dist_config=None,
    logger_: Optional[object] = None,
) -> List[int]:
    local_logger = logger_ or _LOGGER

    env_sizes_str = os.getenv("DLINFER_ASCEND_GRAPH_CAPTURE_SIZES", "")
    if env_sizes_str:
        try:
            env_sizes = [int(x.strip()) for x in env_sizes_str.split(",") if x.strip()]
            if env_sizes:
                local_logger.info(
                    "Using custom ACL graph capture sizes from environment: %s",
                    env_sizes,
                )
                validated_sizes = _collect_positive(env_sizes)
                if validated_sizes:
                    if cache_config is not None:
                        local_logger.info(
                            "Using custom ACL graph batches: %s (max_batches=%s)",
                            validated_sizes,
                            getattr(cache_config, "max_batches", "?"),
                        )
                    return validated_sizes
        except (ValueError, AttributeError) as exc:
            local_logger.warning(
                "Failed to parse DLINFER_ASCEND_GRAPH_CAPTURE_SIZES=%s: %s. Using default sizes.",
                env_sizes_str,
                exc,
            )

    sanitized_original = _collect_positive(original_sizes)
    limited = limit_capture_buckets(
        sanitized_original,
        model_config,
        dist_config,
        max_capture_graphs=DEFAULT_MAX_CAPTURE_GRAPHS,
    )

    if cache_config is not None and limited != sanitized_original:
        local_logger.info(
            "Adjusted ACL graph batches %s → %s (max_batches=%s)",
            sanitized_original,
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

