"""
图分割器：按splitting_ops分割FX graph
完全参考vLLM的split_graph实现
"""
import torch
import torch.fx as fx
from typing import List
from dataclasses import dataclass
from lmdeploy.utils import get_logger

logger = get_logger('dlinfer.graph_splitter')

@dataclass
class SplitItem:
    """子图信息（与vLLM保持一致）"""
    submod_name: str
    graph_id: int
    is_splitting_graph: bool  # 是否是splitting op（attention）
    graph: fx.GraphModule

def split_graph(
    graph: fx.GraphModule,
    ops: List[str]
) -> tuple[fx.GraphModule, List[SplitItem]]:
    """
    按splitting_ops分割图
    
    参考：vLLM的split_graph实现
    /vllm-workspace/vllm/vllm/compilation/backends.py:235-280
    
    Args:
        graph: 原始GraphModule
        ops: splitting op列表，如 ['dlinfer_piecewise.paged_decode_attention']
    
    Returns:
        (split_gm, split_items):
            - split_gm: 分割后的GraphModule，包含多个sub-module
            - split_items: 子图信息列表
    """
    logger.info(f"Splitting graph with ops: {ops}")
    
    # Step 1: 为每个节点分配subgraph_id
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []  # 记录哪些subgraph_id是splitting op
    
    for node in graph.graph.nodes:
        # 跳过特殊节点
        if node.op in ("output", "placeholder"):
            continue
        
        # Debug: 记录所有 call_function 节点的 target
        if node.op == 'call_function':
            target_str = str(node.target)
            if 'attention' in target_str.lower() or 'fill_kv' in target_str.lower():
                logger.info(f"Found potential attention op: {node.op} {target_str}")
        
        # 检查是否是splitting op
        if node.op == 'call_function' and str(node.target) in ops:
            logger.info(f"Matched splitting op: {node.target}")
            # Splitting op前面的节点是一个子图
            subgraph_id += 1
            # Splitting op自己单独是一个子图
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            # 后续节点开始新的子图
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id
    
    logger.debug(f"Total subgraphs: {subgraph_id + 1}, "
                 f"splitting graphs: {split_op_graphs}")
    
    # Step 2: 使用torch.fx的split_module进行实际分割
    # keep_original_order=True 非常重要！
    # 保证有mutation时语义不变
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True
    )
    
    # Step 3: 收集分割后的子模块信息
    outputs = []
    names = [name for (name, module) in split_gm.named_modules()]
    
    for name in names:
        if "." in name or name == "":
            # 跳过递归子模块或根模块
            continue
        
        module = getattr(split_gm, name)
        graph_id = int(name.replace("submod_", ""))
        
        outputs.append(SplitItem(
            submod_name=name,
            graph_id=graph_id,
            is_splitting_graph=(graph_id in split_op_graphs),
            graph=module
        ))
    
    # Step 4: 按graph_id排序
    outputs.sort(key=lambda x: x.graph_id)
    
    logger.info(f"Graph split into {len(outputs)} submodules:")
    for item in outputs:
        logger.debug(f"  {item.submod_name}: "
                     f"{'[ATTENTION]' if item.is_splitting_graph else '[COMPUTE]'}")
    
    return split_gm, outputs

