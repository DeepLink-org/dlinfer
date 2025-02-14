import copy
import torch
import json
import time
from typing import List, Dict, Set
from collections import OrderedDict, defaultdict
from functools import lru_cache

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


class GraphOpearation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "graphOpearation")
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
            if not isinstance(node, GraphOpearation):
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
    def _collect_reshape_nodes(self, graph: Graph) -> tuple[dict, dict, dict]:
        view_replace = {}
        unsqueeze_replace = {}
        squeeze_replace = {}

        for name in list(graph.nodes.keys()):
            node = graph.nodes[name]
            if node.op_type == "viewOperation":
                view_replace[node.op_name] = node
                del graph.nodes[name]
            elif node.op_type == "unsqueezeOperation":
                unsqueeze_replace[node.op_name] = node
                del graph.nodes[name]
            elif node.op_type == "squeezeOperation":
                squeeze_replace[node.op_name] = node
                del graph.nodes[name]

        return view_replace, unsqueeze_replace, squeeze_replace

    def _process_reshape_chain(
        self, input_name: str, reshape_dict: dict, get_reshape_info=None
    ) -> tuple[str, dict]:
        if not input_name in reshape_dict:
            return input_name, None

        reshape_info = (
            get_reshape_info(reshape_dict[input_name])
            if get_reshape_info
            else reshape_dict[input_name].target_reshape_info
        )
        target_name = reshape_dict[input_name].inputs[0]

        while target_name in reshape_dict:
            input_name = target_name
            target_name = reshape_dict[input_name].inputs[0]
            if get_reshape_info:
                reshape_info["dim"].extend(
                    get_reshape_info(reshape_dict[target_name])["dim"]
                )

        if get_reshape_info:
            reshape_info["dim"] = list(reversed(reshape_info["dim"]))

        return target_name, reshape_info

    def _process_node_inputs(
        self, node, view_replace: dict, unsqueeze_replace: dict, squeeze_replace: dict
    ) -> tuple[bool, dict]:
        need_reshape_input = False
        reshape_inputs = {}

        for idx, input_name in enumerate(node.inputs):
            target_name = input_name
            reshape_info = None

            if input_name in view_replace:
                target_name, reshape_info = self._process_reshape_chain(
                    input_name, view_replace
                )

            elif input_name in unsqueeze_replace:
                target_name, reshape_info = self._process_reshape_chain(
                    input_name, unsqueeze_replace, lambda x: x.target_reshape_info
                )

            elif input_name in squeeze_replace:
                target_name, reshape_info = self._process_reshape_chain(
                    input_name, squeeze_replace, lambda x: x.target_reshape_info
                )
                if isinstance(
                    node.inputs, torch.fx.immutable_collections.immutable_list
                ):
                    node.inputs = list(node.inputs)

            if target_name != input_name:
                node.inputs[idx] = target_name
                if reshape_info:
                    reshape_inputs[idx] = reshape_info
                    need_reshape_input = True

        return need_reshape_input, reshape_inputs

    def _update_node_reshape_inputs(
        self, node, need_reshape_input: bool, reshape_inputs: dict
    ):
        node.has_reshape_inputs = need_reshape_input
        node.reshape_inputs = []
        if need_reshape_input:
            for idx, _ in enumerate(node.inputs):
                if idx in reshape_inputs:
                    node.reshape_inputs.append(reshape_inputs[idx])
                else:
                    node.reshape_inputs.append({"reshapeType": "None"})

    def _process_graph_operation_outputs(
        self,
        node: GraphOpearation,
        view_replace: dict,
        unsqueeze_replace: dict,
        squeeze_replace: dict,
    ):
        for idx, output in enumerate(node.outputs):
            target_name = output

            if output in view_replace:
                target_name, _ = self._process_reshape_chain(output, view_replace)
            elif output in unsqueeze_replace:
                target_name, _ = self._process_reshape_chain(output, unsqueeze_replace)
            elif output in squeeze_replace:
                target_name, _ = self._process_reshape_chain(output, squeeze_replace)

            if target_name != output:
                node.outputs[idx] = target_name

    def run(self, graph: Graph, context: dict) -> Graph:
        view_replace, unsqueeze_replace, squeeze_replace = self._collect_reshape_nodes(
            graph
        )
        for name, node in graph.nodes.items():
            if not isinstance(node, GraphOpearation):
                need_reshape_input, reshape_inputs = self._process_node_inputs(
                    node, view_replace, unsqueeze_replace, squeeze_replace
                )
                self._update_node_reshape_inputs(
                    node, need_reshape_input, reshape_inputs
                )
            else:
                self._process_graph_operation_outputs(
                    node, view_replace, unsqueeze_replace, squeeze_replace
                )
        return graph


class GraphOperationPass(OptimizationPass):
    def _collect_graph_nodes(self, graph: Graph) -> List[GraphOpearation]:
        return [
            node for node in graph.nodes.values() if isinstance(node, GraphOpearation)
        ]

    def _process_node_tensors(
        self, node: Operation
    ) -> tuple[List[str], List[str], List[str]]:
        inputs = list(node.inputs)
        outputs = list(node.outputs)
        hosts = list(node.host_inputs) if node.has_host_inputs else []
        return inputs, outputs, hosts

    def _handle_inplace_outputs(self, node: Operation) -> Dict[str, str]:
        inplace_tensors = {}
        if node.has_inplace_output:
            for item in node.inplace_outputs:
                output_idx = item["output_index"]
                input_idx = item["input_index"]
                inplace_tensors[node.outputs[output_idx]] = node.inputs[input_idx]
        return inplace_tensors

    def _update_graph_node(
        self,
        graph_node: GraphOpearation,
        hosts: List[str],
        inplace_tensors: Dict[str, str],
    ):
        graph_node.node_names = list(graph_node.nodes.keys())

        if graph_node.has_infer_shape:
            infer_shape = []
            for node_id, tensor_id in graph_node.infer_shape["value"]:
                node_name = graph_node.node_names[node_id]
                input_name = graph_node.nodes[node_name].inputs[tensor_id]
                infer_shape.append(graph_node.inputs.index(input_name))
            graph_node.infer_shape = {"type": "equal", "value": infer_shape}

        if hosts:
            graph_node.has_host_inputs = True
            graph_node.host_inputs = hosts

        if inplace_tensors:
            graph_node.has_inplace_output = True
            for output_name, input_name in inplace_tensors.items():
                if output_name not in graph_node.outputs:
                    graph_node.outputs.append(output_name)
                    if output_name in graph_node.internals:
                        graph_node.internals.remove(output_name)
                output_idx = graph_node.outputs.index(output_name)
                input_idx = graph_node.inputs.index(input_name)
                graph_node.add_inplace_output(output_idx, input_idx)

    def _process_graph_node(self, graph: Graph, graph_node: GraphOpearation) -> None:
        all_inputs, all_outputs, all_hosts = [], [], []
        inplace_tensors = {}

        for node_name in graph_node.node_names:
            if node_name not in graph.nodes:
                continue

            node = graph.nodes[node_name]
            graph_node.nodes[node_name] = node
            del graph.nodes[node_name]

            inputs, outputs, hosts = self._process_node_tensors(node)
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            all_hosts.extend(hosts)

            if node.special_constants_map:
                graph.update_special_constants(node.special_constants_map)

            inplace_tensors.update(self._handle_inplace_outputs(node))

        all_inputs = list(dict.fromkeys(all_inputs))
        all_outputs = list(dict.fromkeys(all_outputs))
        all_hosts = list(dict.fromkeys(all_hosts))

        graph_inputs = [x for x in all_inputs if x not in all_outputs]
        graph_internals = [
            x
            for x in all_outputs
            if x not in graph_inputs and x not in graph_node.outputs
        ]

        graph_node.set_inputs(graph_inputs)
        graph_node.set_outputs(graph_node.outputs + graph_internals)
        graph_node.set_internals([])

        self._update_graph_node(graph_node, all_hosts, inplace_tensors)

    def run(self, graph: Graph, context: dict) -> Graph:
        for graph_node in self._collect_graph_nodes(graph):
            self._process_graph_node(graph, graph_node)

        return graph


class OptGraph:
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
        node_inputs_count = {input_name: 0 for input_name in input_names}
        node_hosts = []

        for node_id, node in enumerate(graph.nodes.values()):
            for tensor_id, tensor_name in enumerate(node.inputs):
                if tensor_name in host_tensors and node.has_host_inputs:
                    node_hosts.append(
                        {
                            "nodeId": node_id,
                            "tensorId": tensor_id,
                            "tensorName": tensor_name,
                        }
                    )
                if tensor_name in node_inputs_count:
                    node_inputs_count[tensor_name] += 1

        used_inputs = [
            input_name for input_name, count in node_inputs_count.items() if count > 0
        ]

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
    optimizer = OptGraph(graph, input_names, output_names)
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
