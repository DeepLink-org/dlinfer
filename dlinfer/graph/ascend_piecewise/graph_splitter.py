"""
Graph splitter for separating FX graphs by splitting operations.
Based on vLLM split_graph implementation.
"""

import torch
import torch.fx as fx
from typing import List
from dataclasses import dataclass
from lmdeploy.utils import get_logger

logger = get_logger("dlinfer.graph_splitter")


def is_debug_enabled() -> bool:
    """Check if FX graph debugging is enabled via environment variable."""
    import os

    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


@dataclass
class SplitItem:
    """Subgraph information."""

    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(
    graph: fx.GraphModule, ops: List[str]
) -> tuple[fx.GraphModule, List[SplitItem]]:
    """
    Split graph by splitting operations.

    Args:
        graph: Original GraphModule
        ops: List of splitting operations

    Returns:
        (split_gm, split_items): Split GraphModule and subgraph information
    """
    if is_debug_enabled():
        logger.info(f"Splitting graph with ops: {ops}")

    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        if node.op == "call_function" and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []
    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            continue

        module = getattr(split_gm, name)
        graph_id = int(name.replace("submod_", ""))

        outputs.append(
            SplitItem(
                submod_name=name,
                graph_id=graph_id,
                is_splitting_graph=(graph_id in split_op_graphs),
                graph=module,
            )
        )

    outputs.sort(key=lambda x: x.graph_id)

    if is_debug_enabled():
        logger.info(f"Graph split into {len(outputs)} submodules:")
        for item in outputs:
            logger.debug(
                f"  {item.submod_name}: "
                f"{'[ATTENTION]' if item.is_splitting_graph else '[COMPUTE]'}"
            )

    return split_gm, outputs
