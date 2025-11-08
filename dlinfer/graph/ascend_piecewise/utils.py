"""
FX Graph debugging utilities for dlinfer ascend piecewise compilation.
"""

import os
import torch.fx as fx
from typing import List, Dict, Any, Optional
from collections import Counter
from lmdeploy.utils import get_logger
from .graph_splitter import SplitItem

logger = get_logger("dlinfer.debug")


def is_fx_graph_debug_enabled() -> bool:
    """Check if FX graph debugging is enabled via environment variable."""
    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


def debug_fx_graph_structure(gm: fx.GraphModule, title: str = "FX Graph") -> None:
    """
    Debug utility to print comprehensive FX graph structure.

    Args:
        gm: FX GraphModule to debug
        title: Title for the debug output
    """

    logger.info("=" * 80)
    logger.info(f"{title} - Structure Analysis")
    logger.info("=" * 80)

    # Basic graph statistics
    total_nodes = len(list(gm.graph.nodes))
    node_ops = [node.op for node in gm.graph.nodes]
    op_counts = Counter(node_ops)

    logger.info(f"Total nodes: {total_nodes}")
    logger.info(f"Node operations distribution: {dict(op_counts)}")

    # Module and function analysis
    call_functions = [node for node in gm.graph.nodes if node.op == "call_function"]
    call_modules = [node for node in gm.graph.nodes if node.op == "call_module"]

    logger.info(f"Call functions: {len(call_functions)}")
    logger.info(f"Call modules: {len(call_modules)}")

    # Input/Output analysis
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    outputs = [node for node in gm.graph.nodes if node.op == "output"]

    logger.info(f"Input placeholders: {len(placeholders)}")
    logger.info(f"Output nodes: {len(outputs)}")

    if placeholders:
        logger.info("Input details:")
        for i, node in enumerate(placeholders):
            logger.info(f"  [{i}] {node.name}: {node.meta}")

    logger.info("=" * 80)


def debug_fx_graph_nodes(
    gm: fx.GraphModule,
    filter_ops: Optional[List[str]] = None,
    title: str = "FX Graph Nodes",
) -> None:
    """
    Debug utility to print detailed node information.

    Args:
        gm: FX GraphModule to debug
        filter_ops: Optional list of operations to filter (e.g., ['call_function'])
        title: Title for the debug output
    """

    logger.info("=" * 80)
    logger.info(f"{title} - Detailed Node Analysis")
    logger.info("=" * 80)

    node_count = 0
    for node in gm.graph.nodes:
        if filter_ops and node.op not in filter_ops:
            continue

        node_count += 1
        logger.info(f"Node [{node_count}]: {node.name}")
        logger.info(f"  Op: {node.op}")
        logger.info(f"  Target: {node.target}")

        if hasattr(node.target, "__name__"):
            logger.info(f"  Target name: {node.target.__name__}")

        if isinstance(node.target, str):
            logger.info(f"  Target string: '{node.target}'")

        # Show user-specified meta information
        if node.meta:
            logger.info(f"  Meta: {dict(list(node.meta.items())[:3])}")  # First 3 items

        # Show args and types for call_function
        if node.op == "call_function" and node.args:
            arg_types = [type(arg).__name__ for arg in node.args]
            logger.info(f"  Args types: {arg_types}")

        logger.info("  ---")

    logger.info(f"Total nodes analyzed: {node_count}")
    logger.info("=" * 80)


def debug_graph_splitting(
    split_items: List[SplitItem], split_gm: fx.GraphModule
) -> None:
    """
    Debug utility to analyze graph splitting results.

    Args:
        split_items: List of SplitItem objects from graph splitting
        split_gm: The split GraphModule
    """

    logger.info("=" * 80)
    logger.info("Graph Splitting Analysis")
    logger.info("=" * 80)

    total_submodules = len(split_items)
    attention_submodules = sum(1 for item in split_items if item.is_splitting_graph)
    compute_submodules = total_submodules - attention_submodules

    logger.info(f"Total submodules: {total_submodules}")
    logger.info(f"Attention submodules: {attention_submodules}")
    logger.info(f"Compute submodules: {compute_submodules}")

    # Detailed submodule analysis
    logger.info("\nSubmodule Details:")
    for item in split_items:
        module_type = "ATTENTION" if item.is_splitting_graph else "COMPUTE"
        module = getattr(split_gm, item.submod_name)
        node_count = len(list(module.graph.nodes))

        logger.info(
            f"  [{item.graph_id}] {item.submod_name} ({module_type}) - {node_count} nodes"
        )

    # Graph topology analysis
    logger.info(
        f"\nGraph topology: {split_items[0].graph_id} -> {split_items[-1].graph_id}"
    )
    logger.info("=" * 80)


def debug_submodule_wrapping(
    split_items: List[SplitItem], split_gm: fx.GraphModule
) -> None:
    """
    Debug utility to analyze how submodules are wrapped.

    Args:
        split_items: List of SplitItem objects
        split_gm: The split GraphModule
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

        # Count operations in this submodule
        node_ops = Counter(node.op for node in module.graph.nodes)
        call_functions = [
            node for node in module.graph.nodes if node.op == "call_function"
        ]

        logger.info(f"Submodule: {item.submod_name}")
        logger.info(f"  Graph ID: {item.graph_id}")
        logger.info(f"  Type: {wrapper_type}")
        logger.info(f"  Operations: {dict(node_ops)}")
        logger.info(f"  Call functions: {len(call_functions)}")

        if call_functions:
            targets = [str(node.target) for node in call_functions]
            logger.info(f"  Function targets: {targets[:5]}")  # First 5 targets

        logger.info("  ---")

    logger.info("=" * 80)


def debug_compilation_summary(
    original_gm: fx.GraphModule,
    split_gm: fx.GraphModule,
    split_items: List[SplitItem],
    compilation_count: int,
) -> None:
    """
    Debug utility to print compilation summary.

    Args:
        original_gm: Original input GraphModule
        split_gm: Resulting split GraphModule
        split_items: List of SplitItem objects
        compilation_count: Current compilation call count
    """

    logger.info("=" * 80)
    logger.info(f"Compilation Summary (Call #{compilation_count})")
    logger.info("=" * 80)

    original_nodes = len(list(original_gm.graph.nodes))
    total_split_nodes = sum(
        len(list(getattr(split_gm, item.submod_name).graph.nodes))
        for item in split_items
    )

    logger.info(f"Original graph nodes: {original_nodes}")
    logger.info(f"Split graph total nodes: {total_split_nodes}")
    logger.info(f"Number of submodules: {len(split_items)}")
    logger.info(f"Graph overhead: {total_split_nodes - original_nodes} nodes")

    # Efficiency metrics
    if original_nodes > 0:
        overhead_ratio = (total_split_nodes - original_nodes) / original_nodes * 100
        logger.info(f"Graph splitting overhead: {overhead_ratio:.1f}%")

    logger.info("=" * 80)
