"""Shared helpers and debug utilities for Ascend piecewise graph mode."""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch.fx as fx
from lmdeploy.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .graph_splitter import SplitItem

logger = get_logger("dlinfer.debug")


@lru_cache(maxsize=1)
def is_debug_enabled() -> bool:
    """Return True when verbose debugging for piecewise graph mode is on."""
    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


@lru_cache(maxsize=1)
def is_acl_graph_debug_enabled() -> bool:
    """Dedicated flag for ACL graph level debugging."""
    env = os.environ.get("DLINFER_ASCEND_ACL_GRAPH_DEBUG")
    if env is not None:
        return env == "1"
    return is_debug_enabled()


def is_fx_graph_debug_enabled() -> bool:
    """Check if FX graph debugging is enabled via environment variable."""
    return is_debug_enabled()


def debug_fx_graph_structure(gm: fx.GraphModule, title: str = "FX Graph") -> None:
    """
    Debug utility to print comprehensive FX graph structure.
    """

    logger.info("=" * 80)
    logger.info(f"{title} - Structure Analysis")
    logger.info("=" * 80)

    total_nodes = len(list(gm.graph.nodes))
    node_ops = [node.op for node in gm.graph.nodes]
    op_counts = Counter(node_ops)

    logger.info("Total nodes: %s", total_nodes)
    logger.info("Node operations distribution: %s", dict(op_counts))

    call_functions = [node for node in gm.graph.nodes if node.op == "call_function"]
    call_modules = [node for node in gm.graph.nodes if node.op == "call_module"]
    logger.info("Call functions: %s", len(call_functions))
    logger.info("Call modules: %s", len(call_modules))

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    outputs = [node for node in gm.graph.nodes if node.op == "output"]
    logger.info("Input placeholders: %s", len(placeholders))
    logger.info("Output nodes: %s", len(outputs))

    if placeholders:
        logger.info("Input details:")
        for i, node in enumerate(placeholders):
            logger.info("  [%s] %s: %s", i, node.name, node.meta)

    logger.info("=" * 80)


def debug_fx_graph_nodes(
    gm: fx.GraphModule,
    filter_ops: Optional[List[str]] = None,
    title: str = "FX Graph Nodes",
) -> None:
    """
    Debug utility to print detailed node information.
    """

    logger.info("=" * 80)
    logger.info(f"{title} - Detailed Node Analysis")
    logger.info("=" * 80)

    node_count = 0
    for node in gm.graph.nodes:
        if filter_ops and node.op not in filter_ops:
            continue

        node_count += 1
        logger.info("Node [%s]: %s", node_count, node.name)
        logger.info("  Op: %s", node.op)
        logger.info("  Target: %s", node.target)

        if hasattr(node.target, "__name__"):
            logger.info("  Target name: %s", node.target.__name__)

        if isinstance(node.target, str):
            logger.info("  Target string: '%s'", node.target)

        if node.meta:
            logger.info("  Meta: %s", dict(list(node.meta.items())[:3]))

        if node.op == "call_function" and node.args:
            arg_types = [type(arg).__name__ for arg in node.args]
            logger.info("  Args types: %s", arg_types)

        logger.info("  ---")

    logger.info("Total nodes analyzed: %s", node_count)
    logger.info("=" * 80)


def debug_graph_splitting(
    split_items: List["SplitItem"], split_gm: fx.GraphModule
) -> None:
    """
    Debug utility to analyze graph splitting results.
    """

    logger.info("=" * 80)
    logger.info("Graph Splitting Analysis")
    logger.info("=" * 80)

    total_submodules = len(split_items)
    attention_submodules = sum(1 for item in split_items if item.is_splitting_graph)
    compute_submodules = total_submodules - attention_submodules

    logger.info("Total submodules: %s", total_submodules)
    logger.info("Attention submodules: %s", attention_submodules)
    logger.info("Compute submodules: %s", compute_submodules)

    logger.info("\nSubmodule Details:")
    for item in split_items:
        module_type = "ATTENTION" if item.is_splitting_graph else "COMPUTE"
        module = getattr(split_gm, item.submod_name)
        node_count = len(list(module.graph.nodes))
        logger.info(
            "  [%s] %s (%s) - %s nodes",
            item.graph_id,
            item.submod_name,
            module_type,
            node_count,
        )

    logger.info(
        "\nGraph topology: %s -> %s",
        split_items[0].graph_id,
        split_items[-1].graph_id,
    )
    logger.info("=" * 80)


def debug_submodule_wrapping(
    split_items: List["SplitItem"], split_gm: fx.GraphModule
) -> None:
    """
    Debug utility to analyze how submodules are wrapped.
    """

    logger.info("=" * 80)
    logger.info("Submodule Wrapping Analysis")
    logger.info("=" * 80)

    for item in split_items:
        wrapper_type = (
            "EagerExecutionWrapper"
            if item.is_splitting_graph
            else "AscendPiecewiseGraphWrapper"
        )
        module = getattr(split_gm, item.submod_name)
        node_ops = Counter(node.op for node in module.graph.nodes)
        call_functions = [
            node for node in module.graph.nodes if node.op == "call_function"
        ]

        logger.info("Submodule: %s", item.submod_name)
        logger.info("  Graph ID: %s", item.graph_id)
        logger.info("  Type: %s", wrapper_type)
        logger.info("  Operations: %s", dict(node_ops))
        logger.info("  Call function nodes: %s", len(call_functions))
        logger.info("  ---")

    logger.info("=" * 80)


def debug_compilation_summary(
    original_gm: fx.GraphModule,
    split_gm: fx.GraphModule,
    split_items: List["SplitItem"],
    compilation_count: int,
) -> None:
    """
    Final debug summary after compilation.
    """

    logger.info("=" * 80)
    logger.info("Compilation Summary")
    logger.info("=" * 80)
    logger.info("Compilation count: %s", compilation_count)
    logger.info("Original graph nodes: %s", len(list(original_gm.graph.nodes)))
    logger.info("Split graph nodes: %s", len(list(split_gm.graph.nodes)))
    logger.info("Submodules: %s", len(split_items))

    for item in split_items:
        logger.info(
            "  [%s] %s -> %s",
            item.graph_id,
            item.submod_name,
            "ATTN" if item.is_splitting_graph else "COMPUTE",
        )

    logger.info("=" * 80)


class PiecewiseEnvConfig:
    """Centralized environment variable configuration for piecewise graph."""

    # Debug flags
    DEBUG_CAPTURE = "DLINFER_ASCEND_DEBUG_CAPTURE"
    PIECEWISE_GRAPH_DEBUG = "DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG"
    ACL_GRAPH_DEBUG = "DLINFER_ASCEND_ACL_GRAPH_DEBUG"

    # Capture configuration
    CAPTURE_SIZES = "DLINFER_ASCEND_CAPTURE_SIZES"
    MAX_CAPTURE_GRAPHS = "DLINFER_ASCEND_MAX_CAPTURE_GRAPHS"
    GRAPH_CAPTURE_SIZES = "DLINFER_ASCEND_GRAPH_CAPTURE_SIZES"

    # Hardware and system configuration
    MULTISTREAM_SHARED_EXPERT = "DLINFER_ASCEND_MULTISTREAM_SHARED_EXPERT"
    HCCL_OP_EXPANSION_MODE = "HCCL_OP_EXPANSION_MODE"

    @staticmethod
    def get_debug_capture() -> bool:
        """Get debug capture flag."""
        return os.environ.get(PiecewiseEnvConfig.DEBUG_CAPTURE, "0") == "1"

    @staticmethod
    def get_capture_sizes() -> Optional[List[int]]:
        """Get capture sizes from environment variable."""
        env_str = os.getenv(PiecewiseEnvConfig.CAPTURE_SIZES)
        if not env_str:
            return None
        try:
            return sorted(
                {
                    int(item.strip())
                    for item in env_str.split(",")
                    if item.strip() and int(item.strip()) > 0
                }
            )
        except ValueError:
            return None

    @staticmethod
    def get_max_capture_graphs() -> Optional[int]:
        """Get max capture graphs from environment variable."""
        env_str = os.getenv(PiecewiseEnvConfig.MAX_CAPTURE_GRAPHS)
        if not env_str:
            return None
        try:
            return int(env_str)
        except ValueError:
            return None

    @staticmethod
    def get_graph_capture_sizes() -> Optional[str]:
        """Get graph capture sizes from environment variable."""
        return os.getenv(PiecewiseEnvConfig.GRAPH_CAPTURE_SIZES, "")

    @staticmethod
    def get_shared_expert_overlap() -> bool:
        """Get shared expert overlap flag."""
        env = os.getenv(PiecewiseEnvConfig.MULTISTREAM_SHARED_EXPERT)
        if not env:
            return False
        try:
            return bool(int(env))
        except ValueError:
            lowered = env.strip().lower()
            return lowered in {"true", "on", "yes"}

    @staticmethod
    def get_hccl_mode() -> str:
        """Get HCCL operation expansion mode."""
        return os.getenv(PiecewiseEnvConfig.HCCL_OP_EXPANSION_MODE, "").upper()


__all__ = [
    "is_debug_enabled",
    "is_acl_graph_debug_enabled",
    "is_fx_graph_debug_enabled",
    "debug_fx_graph_structure",
    "debug_fx_graph_nodes",
    "debug_graph_splitting",
    "debug_submodule_wrapping",
    "debug_compilation_summary",
    "PiecewiseEnvConfig",
]
