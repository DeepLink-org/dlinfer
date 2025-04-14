import copy
from importlib import import_module
from contextlib import contextmanager

import torch
import torch.fx
import torch.fx.traceback
from torch.fx.passes.dialect.common.cse_pass import CSEPass

from dlinfer.graph.dicp.dynamo_bridge.torch_version import is_torch_210_or_higher
from dlinfer.graph.dicp.vendor.AtbGraph.conversion import (
    AtenToAtbTransformer,
    ViewSymIntTransformer,
)
from ...dynamo_bridge.graph import GraphTransformer

if is_torch_210_or_higher:
    from dlinfer.graph.dicp.dynamo_bridge.op_transformer import (
        BackendPatternMatcherTransformer,
    )
    from dlinfer.graph.dicp.vendor.AtbGraph.pattern_replacement import (
        atb_pattern_matcher,
        torch_patterns_cls_list_1,
        torch_patterns_cls_list_2,
        torch_patterns_cls_list_3,
    )


@contextmanager
def preserve_meta_val():
    target_module = import_module("torch.fx.interpreter")
    assert not target_module is None
    target_variable_str = "Transformer"
    target_variable = getattr(target_module, target_variable_str)
    target_method_str = "run_node"
    origin_target_method = getattr(target_variable, target_method_str)

    def new_target_method(obj: torch.fx.Transformer, n: torch.fx.Node):
        proxy: torch.fx.Proxy = origin_target_method(obj, n)
        with obj._set_current_node(n):
            current_meta = torch.fx.traceback.get_current_meta()
            if "val" in current_meta:
                proxy.node.meta["val"] = current_meta["val"]
        return proxy

    setattr(target_variable, target_method_str, new_target_method)
    yield
    setattr(target_variable, target_method_str, origin_target_method)


def atbgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    with preserve_meta_val():
        gm = ViewSymIntTransformer(gm).transform()
        gm.graph.eliminate_dead_code()
        cse_pass = CSEPass()
        cse_pass_result = cse_pass(gm)
        gm = cse_pass_result.graph_module

    gm = BackendPatternMatcherTransformer(
        atb_pattern_matcher, torch_patterns_cls_list_1
    ).transform(gm)

    gm = AtenToAtbTransformer(gm).transform()

    # For bug in pytorch
    # Avoid for dynamic shape
    GraphTransformer.infer_shape_dtype(gm)
    return gm
