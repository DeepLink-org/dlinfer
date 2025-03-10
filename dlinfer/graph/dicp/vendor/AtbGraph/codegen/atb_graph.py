import copy
import torch
import json
from typing import List, Dict, Set, Tuple, Optional, Iterable
from collections import OrderedDict, defaultdict
from functools import lru_cache
import contextlib

from dlinfer.graph.dicp.vendor.AtbGraph.codegen import atb_infer_param as infer_param
from dlinfer.graph.dicp.dynamo_bridge.utils import process_sym_name
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.utils import (
    AclDataType,
    AclFormat,
    get_acl_dtype,
    get_torch_dtype,
)


def get_shape(elem):
    if hasattr(elem, "meta"):
        elem = elem.meta["val"]
    if isinstance(elem, torch.SymInt) or isinstance(elem, torch.SymBool):
        return [1], 1
    shape = list(elem.shape)
    if len(shape) == 0:
        shape = [1]
    shape = [process_sym_name(dim) for dim in shape]
    dim_num = len(shape)
    return shape, dim_num


def get_dtype(elem):
    if hasattr(elem, "meta"):
        elem = elem.meta["val"]
    if isinstance(elem, torch.SymInt):
        return AclDataType.ACL_INT32.value
    if isinstance(elem, torch.SymBool):
        return AclDataType.ACL_BOOL.value
    return get_acl_dtype(elem.dtype)


class Operation:
    def __init__(self, op_name: str, op_type: str):
        self.op_name = op_name
        self.op_type = op_type
        self.param = {}
        self.inputs = []
        self.outputs = []
        self.has_host_inputs = False
        self.host_inputs = []
        self.has_reshape_inputs = False
        self.reshape_inputs = []
        self.special_constants_map = {}
        self.has_inplace_output = False
        self.inplace_outputs = []

    def add_inplace_output(self, output_idx, input_idx):
        self.inplace_outputs.append(
            {"output_index": int(output_idx), "input_index": int(input_idx)}
        )

    def set_input(self, x):
        self.inputs = x

    def set_output(self, x):
        self.outputs = x

    def set_special_constants(self, x):
        self.special_constants_map = x

    def add_input(self, x):
        self.inputs.append(x)

    def add_output(self, x):
        self.outputs.append(x)

    def add_special_constants(self, param_name, special_call_str):
        self.special_constants_map[param_name] = special_call_str

    def set_param(self, x):
        if not isinstance(x, dict):
            x = infer_param.to_dict(x)
        self.param = x

    def build(self):
        node = {
            "nodeType": "singleOperation",
            "value": {
                "name": self.op_name,
                "type": self.op_type,
                "param": self.param,
                "inputNames": self.inputs,
                "outputNames": self.outputs,
                "hasHostInputs": self.has_host_inputs,
                "hostInputNames": self.host_inputs,
                "hasReshapeInputs": self.has_reshape_inputs,
                "reshapeInputs": self.reshape_inputs,
                "hasInplaceOutputs": self.has_inplace_output,
                "inplaceOutputs": self.inplace_outputs,
            },
        }
        return node


class GetItemOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "getitemOperation")
        self.index = -1


class InplaceOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "inplaceOperation")
        self.input_index = -1
        self.target_index = -1
        self.target = None


class TupleOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "tupleOperation")


class ViewOperation(Operation):
    def __init__(self, name):
        super().__init__(name, "viewOperation")
        self.target_shape = []
        self.target_reshape_info = {}


class UnsqueezeOperation(Operation):
    def __init__(self, name):
        super().__init__(name, "unsqueezeOperation")
        self.dim = []
        self.target_reshape_info = {}


class SqueezeOperation(Operation):
    def __init__(self, name):
        super().__init__(name, "squeezeOperation")
        self.dim = []
        self.target_reshape_info = {}


class GraphOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "GraphOperation")
        self.op_name = name
        self.op_type = "graphOperation"
        self.nodes: OrderedDict[str, Operation] = OrderedDict()
        self.inputs = []
        self.outputs = []
        self.internals = []
        self.node_size = -1
        self.node_names = []
        self.has_host_inputs = False
        self.host_inputs = []
        self.has_infer_shape = False
        self.infer_shape = ""
        self.has_inplace_output = False
        self.inplace_outputs = []

    def add_inplace_output(self, output_idx, input_idx):
        self.inplace_outputs.append(
            {"output_index": int(output_idx), "input_index": int(input_idx)}
        )

    def set_node_names(self, x):
        self.node_names = x

    def add_node_name(self, x):
        self.node_names.append(x)

    def set_inputs(self, x):
        self.inputs = x

    def set_outputs(self, x):
        self.outputs = x

    def set_internals(self, x):
        self.internals = x

    def set_nodes(self, x):
        self.nodes = x

    def add_input(self, x):
        self.inputs.append(x)

    def add_output(self, x):
        self.outputs.append(x)

    def add_internal(self, x):
        self.internals.append(x)

    def add_node(self, x):
        self.nodes.append(x)

    def build(self):
        graph = {
            "nodeType": "graphOperation",
            "value": {
                "nodes": [node.build() for _, node in self.nodes.items()],
                "inputNames": self.inputs,
                "outputNames": self.outputs,
                "internalNames": self.internals,
                "nodeSize": len(self.nodes),
                "hasHostInputs": self.has_host_inputs,
                "hostInputNames": self.host_inputs,
                "hasInferShape": self.has_infer_shape,
                "inferShape": self.infer_shape,
                "hasInplaceOutputs": self.has_inplace_output,
                "inplaceOutputs": self.inplace_outputs,
            },
        }
        return graph


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.internals = []
        self.special_constants_map: OrderedDict[str, str] = OrderedDict()
        self.nodes: OrderedDict[str, Operation] = OrderedDict()
        self.hosts = []

    def set_hosts(self, x):
        self.hosts = x

    def set_inputs(self, x):
        self.inputs = x

    def set_outputs(self, x):
        self.outputs = x

    def set_internals(self, x):
        self.internals = x

    def set_special_constants(self, x):
        self.special_constants_map = x

    def add_input(self, x):
        self.inputs.append(x)

    def add_output(self, x):
        self.outputs.append(x)

    def add_internals(self, x):
        self.internals.append(x)

    def extend_inputs(self, x):
        self.inputs += x

    def update_special_constants(self, x):
        self.special_constants_map.update(x)

    def add_node(self, x):
        self.nodes[x.op_name] = x

    def to_json(self):
        atb_graph = {
            "name": self.name,
            "inputNames": self.inputs,
            "outputNames": self.outputs,
            "internalNames": self.internals,
            "nodes": [node for _, node in self.nodes.items()],
            "hostTensorNames": self.hosts,
            "nodeSize": len(self.nodes),
        }
        return json.dumps(atb_graph, sort_keys=True)


def get_input_data_node(node_list, node_name):
    for node in node_list:
        if node.name == node_name:
            return node
    return None


def make_output_tensor_desc(
    output_names,
    output_data_nodes,
):
    output_tensor_descs = {"param": {}, "create": {}}

    def process_node(output_name, node, input_name=None, need_reshape=False):
        dims, dim_num = get_shape(node)
        dims_str = f'[{",".join(dims)}]'
        dtype = get_dtype(node)
        info = f""" {{"format": {AclFormat.ACL_FORMAT_ND.value}, "dtype": {dtype}, "dimNum": {dim_num}, "dims": {dims_str} }} """

        output_tensor_descs["param"][output_name] = info
        output_tensor_descs["create"][output_name] = {
            "dtype": str(get_torch_dtype(dtype)),
            "shape": dims_str,
            "input": input_name,
            "need_reshape": need_reshape,
        }

    for idx, output in enumerate(output_data_nodes):
        output_name = output_names[idx]
        process_node(output_name, output)

    return output_tensor_descs


class OptimizationPass:
    def run(self, graph: Graph, context: dict) -> Graph:
        raise NotImplementedError

    def _validate(self, graph: Graph):
        pass


class TuplePass(OptimizationPass):
    def run(self, graph: Graph, context: dict) -> Graph:
        tuple_replace = {}
        for name in list(graph.nodes.keys()):
            node = graph.nodes[name]
            if isinstance(node, TupleOperation):
                op_name = node.op_name
                for idx, input in enumerate(node.inputs):
                    key_name = f"{op_name}__{idx}"
                    tuple_replace[key_name] = input
                del graph.nodes[name]

        context["tuple_replace"] = tuple_replace
        return graph


class GetItemPass(OptimizationPass):
    def run(self, graph: Graph, context: dict) -> Graph:
        getitem_replace = {}
        tuple_replace = context.get("tuple_replace", {})

        for name in list(graph.nodes.keys()):
            node = graph.nodes[name]
            if isinstance(node, GetItemOperation):
                real_name = f"{node.inputs[0]}__{node.index}"
                if real_name in tuple_replace:
                    real_name = tuple_replace[real_name]
                if real_name in getitem_replace:
                    real_name = getitem_replace[real_name]
                getitem_replace[node.outputs[0]] = real_name
                del graph.nodes[name]

        for name in graph.nodes.keys():
            node = graph.nodes[name]
            if not isinstance(node, GraphOperation):
                for idx, input in enumerate(node.inputs):
                    if input in getitem_replace:
                        node.inputs[idx] = getitem_replace[input]
            else:
                for idx, output in enumerate(node.outputs):
                    if output in getitem_replace:
                        node.outputs[idx] = getitem_replace[output]

        for idx, output in enumerate(context["output_names"]):
            if output in getitem_replace:
                context["output_names"][idx] = getitem_replace[output]
        context["getitem_replace"] = getitem_replace
        return graph


class ReshapePass(OptimizationPass):
    def _collect_reshape_nodes(self, graph: Graph) -> Tuple[Dict[str, Operation], ...]:
        reshape_ops = {
            "viewOperation": {},
            "unsqueezeOperation": {},
            "squeezeOperation": {},
        }

        for name in list(graph.nodes.keys()):
            node = graph.nodes[name]
            if node.op_type in reshape_ops:
                reshape_ops[node.op_type][name] = node
                del graph.nodes[name]

        return tuple(reshape_ops.values())

    def _process_reshape_chain(
        self, input_name: str, reshape_nodes: dict, get_reshape_info=None
    ) -> Tuple[str, Optional[Dict]]:
        if input_name not in reshape_nodes:
            return input_name, None

        current_node = reshape_nodes[input_name]
        reshape_info = (
            get_reshape_info(current_node)
            if get_reshape_info
            else current_node.target_reshape_info
        )

        target_name = current_node.inputs[0]
        while target_name in reshape_nodes:
            current_node = reshape_nodes[target_name]
            if get_reshape_info and "dim" in reshape_info:
                reshape_info["dim"].extend(
                    get_reshape_info(current_node).get("dim", [])
                )
            target_name = current_node.inputs[0]

        if get_reshape_info and "dim" in reshape_info:
            reshape_info["dim"].reverse()

        return target_name, reshape_info

    def _process_node_inputs(
        self, node, view_replace: dict, unsqueeze_replace: dict, squeeze_replace: dict
    ) -> tuple[bool, dict]:
        reshape_handlers = {
            "view": (view_replace, None),
            "unsqueeze": (unsqueeze_replace, lambda x: x.target_reshape_info),
            "squeeze": (squeeze_replace, lambda x: x.target_reshape_info),
        }

        need_reshape = False
        reshape_inputs = {}

        for idx, input_name in enumerate(node.inputs):
            for op_type, (replace_dict, info_getter) in reshape_handlers.items():
                if input_name not in replace_dict:
                    continue

                target, info = self._process_reshape_chain(
                    input_name, replace_dict, info_getter
                )
                if target != input_name:
                    node.inputs[idx] = target
                    if info:
                        reshape_inputs[idx] = info
                        need_reshape = True

                if op_type == "squeeze" and isinstance(node.inputs, tuple):
                    node.inputs = list(node.inputs)
                break

        return need_reshape, reshape_inputs

    def _update_node_reshape_inputs(
        self, node, need_reshape: bool, reshape_inputs: dict
    ):
        node.has_reshape_inputs = need_reshape
        node.reshape_inputs = (
            [
                reshape_inputs.get(idx, {"reshapeType": "None"})
                for idx in range(len(node.inputs))
            ]
            if need_reshape
            else []
        )

    def run(self, graph: Graph, context: dict) -> Graph:
        view, unsqueeze, squeeze = self._collect_reshape_nodes(graph)

        for node in graph.nodes.values():
            if isinstance(node, GraphOperation):
                self._process_graph_outputs(node, view, unsqueeze, squeeze)
            else:
                need_reshape, reshape_map = self._process_node_inputs(
                    node, view, unsqueeze, squeeze
                )
                self._update_node_reshape_inputs(node, need_reshape, reshape_map)

        return graph

    def _process_graph_outputs(
        self,
        node: GraphOperation,
        view: Dict[str, Operation],
        unsqueeze: Dict[str, Operation],
        squeeze: Dict[str, Operation],
    ):
        replace_dicts = (view, unsqueeze, squeeze)
        for idx, output in enumerate(node.outputs):
            for replace_dict in replace_dicts:
                if output in replace_dict:
                    target, _ = self._process_reshape_chain(output, replace_dict)
                    if target != output:
                        node.outputs[idx] = target
                    break


class GraphOperationPass(OptimizationPass):
    def _collect_graph_nodes(self, graph: Graph) -> List[GraphOperation]:
        return list(n for n in graph.nodes.values() if isinstance(n, GraphOperation))

    def _process_node_tensors(
        self, node: Operation
    ) -> Tuple[List[str], List[str], List[str]]:
        return (
            node.inputs.copy(),
            node.outputs.copy(),
            node.host_inputs.copy() if node.has_host_inputs else [],
        )

    def _handle_inplace_outputs(self, node: Operation) -> Dict[str, str]:
        if not node.has_inplace_output:
            return {}
        return {
            node.outputs[item.get("output_index", -1)]: node.inputs[
                item.get("input_index", -1)
            ]
            for item in node.inplace_outputs
            if item.get("output_index") is not None
            and item.get("input_index") is not None
        }

    def _update_infer_shape(self, graph_node: GraphOperation):
        if not graph_node.has_infer_shape:
            return

        graph_node.infer_shape = {
            "type": "equal",
            "value": [
                self._get_input_index(graph_node, node_id, tensor_id)
                for node_id, tensor_id in graph_node.infer_shape["value"]
            ],
        }

    def _get_input_index(
        self, graph_node: GraphOperation, node_id: int, tensor_id: int
    ) -> int:
        node_name = graph_node.node_names[node_id]
        input_name = graph_node.nodes[node_name].inputs[tensor_id]
        try:
            return graph_node.inputs.index(input_name)
        except ValueError:
            return -1

    def _update_graph_node(
        self,
        graph_node: GraphOperation,
        hosts: List[str],
        inplace_tensors: Dict[str, str],
    ):
        graph_node.node_names = list(graph_node.nodes.keys())
        self._update_infer_shape(graph_node)

        if host_list := self._unique(hosts):
            graph_node.has_host_inputs = True
            graph_node.host_inputs = host_list

        if inplace_tensors:
            self._process_inplace_tensors(graph_node, inplace_tensors)

    def _process_inplace_tensors(
        self, graph_node: GraphOperation, inplace_tensors: Dict[str, str]
    ):
        graph_node.has_inplace_output = True
        for output_name, input_name in inplace_tensors.items():
            self._update_output_list(graph_node, output_name)
            self._add_inplace_mapping(graph_node, output_name, input_name)

    def _update_output_list(self, graph_node: GraphOperation, output_name: str):
        if output_name not in graph_node.outputs:
            graph_node.outputs.append(output_name)
            with contextlib.suppress(ValueError):
                graph_node.internals.remove(output_name)

    def _add_inplace_mapping(
        self, graph_node: GraphOperation, output_name: str, input_name: str
    ):
        output_idx = graph_node.outputs.index(output_name)
        input_idx = graph_node.inputs.index(input_name)
        graph_node.add_inplace_output(output_idx, input_idx)

    def _process_graph_node(self, graph: Graph, graph_node: GraphOperation) -> None:
        accumulators = defaultdict(list)
        inplace_tensors = {}

        for node_name in graph_node.node_names:
            if (node := graph.nodes.pop(node_name, None)) is None:
                continue

            self._process_single_node(
                graph, graph_node, node, accumulators, inplace_tensors
            )

        input_set = set(accumulators["outputs"])
        graph_inputs = [
            x for x in self._unique(accumulators["inputs"]) if x not in input_set
        ]

        self._finalize_graph_node(
            graph_node, accumulators, graph_inputs, inplace_tensors
        )

    def _process_single_node(
        self, graph, graph_node, node, accumulators, inplace_tensors
    ):
        graph_node.nodes[node.op_name] = node
        inputs, outputs, hosts = self._process_node_tensors(node)

        accumulators["inputs"].extend(inputs)
        accumulators["outputs"].extend(outputs)
        accumulators["hosts"].extend(hosts)

        if node.special_constants_map:
            graph.special_constants_map.update(node.special_constants_map)

        inplace_tensors.update(self._handle_inplace_outputs(node))

    def _finalize_graph_node(
        self, graph_node, accumulators, graph_inputs, inplace_tensors
    ):
        graph_internals = [
            x
            for x in self._unique(accumulators["outputs"])
            if x not in graph_inputs and x not in graph_node.outputs
        ]

        graph_node.set_inputs(graph_inputs)
        graph_node.set_outputs(graph_node.outputs + graph_internals)
        graph_node.set_internals([])
        self._update_graph_node(graph_node, accumulators["hosts"], inplace_tensors)

    @staticmethod
    def _unique(items: Iterable) -> List:
        return list(dict.fromkeys(items))

    def run(self, graph: Graph, context: dict) -> Graph:
        for graph_node in self._collect_graph_nodes(graph):
            self._process_graph_node(graph, graph_node)
        return graph


class GraphOptimizer:
    def __init__(
        self, original_graph: Graph, input_names: List[str], output_names: List[str]
    ):
        self.original_graph = original_graph
        self.input_names = input_names.copy()
        self.output_names = output_names.copy()
        self.passes = [TuplePass(), GetItemPass(), ReshapePass(), GraphOperationPass()]
        self.context = defaultdict(dict)
        self.context.update(
            {
                "input_names": self.input_names,
                "output_names": self.output_names,
            }
        )

    def optimize(self) -> Graph:
        processed_graph = copy.deepcopy(self.original_graph)
        for pass_instance in self.passes:
            processed_graph = pass_instance.run(processed_graph, self.context)
        return self._finalize_graph(processed_graph)

    def _collect_tensors(self, graph: Graph) -> tuple[Set[str], Set[str]]:
        all_tensors = set()
        host_tensors = set()

        for node in graph.nodes.values():
            all_tensors.update(node.inputs)
            all_tensors.update(node.outputs)

            if node.has_host_inputs:
                host_tensors.update(node.host_inputs)

            if node.special_constants_map:
                graph.update_special_constants(node.special_constants_map)

        return all_tensors, host_tensors

    def _process_node_inputs(
        self, graph: Graph, input_names: List[str], host_tensors: Set[str]
    ) -> tuple[List[Dict], List[str]]:
        node_inputs_count = defaultdict(int)
        node_hosts = []

        host_tensors_set = frozenset(host_tensors)

        for node_id, node in enumerate(graph.nodes.values()):
            for tensor_id, tensor_name in enumerate(node.inputs):
                if tensor_name in host_tensors_set and node.has_host_inputs:
                    node_hosts.append(
                        {
                            "nodeId": node_id,
                            "tensorId": tensor_id,
                            "tensorName": tensor_name,
                        }
                    )
                node_inputs_count[tensor_name] += 1

        used_inputs = [k for k in input_names if node_inputs_count[k] > 0]
        return node_hosts, used_inputs

    def _categorize_tensors(
        self,
        all_tensors: Set[str],
        node_inputs: List[str],
        output_names: List[str],
        special_constants: Dict,
    ) -> tuple[List[str], List[str]]:
        node_outputs = list(output_names)
        node_internals = []

        for tensor in all_tensors:
            if (
                tensor not in node_inputs
                and tensor not in node_outputs
                and tensor not in special_constants
            ):
                node_internals.append(tensor)

        return node_outputs, node_internals

    def _finalize_graph(self, graph: Graph) -> Graph:
        input_names = self.context.get("input_names", [])
        output_names = self.context.get("output_names", [])

        all_tensors, host_tensors = self._collect_tensors(graph)

        node_hosts, node_inputs = self._process_node_inputs(
            graph, input_names, host_tensors
        )

        node_outputs, node_internals = self._categorize_tensors(
            all_tensors, node_inputs, output_names, graph.special_constants_map
        )

        graph.set_inputs(node_inputs)
        graph.set_outputs(node_outputs)
        graph.set_internals(node_internals)
        graph.set_hosts(node_hosts)

        for name in graph.nodes:
            graph.nodes[name] = graph.nodes[name].build()

        return graph


def parse_graph(
    graph: Graph,
    input_names,
    output_names,
    input_data_nodes,
    output_data_nodes,
    py_output_names,
):
    optimizer = GraphOptimizer(graph, input_names, output_names)
    optimized_graph = optimizer.optimize()

    output_tensor_descs = make_output_tensor_desc(
        optimizer.context.get("output_names", {}),
        output_data_nodes,
    )

    getitem_replace = optimizer.context.get("getitem_replace", {})
    for idx, tensor in enumerate(py_output_names):
        if tensor in getitem_replace:
            py_output_names[idx] = getitem_replace[tensor]

    optimized_graph.extend_inputs(optimized_graph.special_constants_map.keys())

    return optimized_graph, output_tensor_descs, py_output_names
