import dataclasses
import json
import torch
from collections import OrderedDict
from dlinfer.graph.dicp.vendor.AtbGraph.codegen import atb_infer_param as infer_param
from dlinfer.graph.dicp.dynamo_bridge.utils import process_sym_name
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.utils import (
    AclDataType,
    AclFormat,
    get_acl_dtype,
    get_torch_dtype,
)
from typing import Dict, List, Any, Tuple, Optional, NamedTuple


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
        self.is_reshape_op = False

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


class TupleOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "tupleOperation")


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


@dataclasses.dataclass
class TensorDesc:
    dtype: str
    shape: str
    input: Optional[str] = None
    need_reshape: bool = False


class TensorInfo(NamedTuple):
    shape_str: str
    dtype_str: str


def _process_shape(shape: Tuple[int, ...]) -> str:
    dims = [process_sym_name(dim) for dim in shape] if shape else ["1"]
    return f'[{",".join(dims)}]'


def get_shape(elem: Any) -> Tuple[List[Any], int]:
    if hasattr(elem, "meta"):
        elem = elem.meta["val"]

    if isinstance(elem, (torch.SymInt, torch.SymBool)):
        return [1], 1

    shape = list(elem.shape) if elem.shape else [1]
    return shape, len(shape)


def get_dtype(elem: Any) -> AclDataType:
    if hasattr(elem, "meta"):
        elem = elem.meta["val"]

    if isinstance(elem, torch.SymInt):
        return AclDataType.ACL_INT32.value
    if isinstance(elem, torch.SymBool):
        return AclDataType.ACL_BOOL.value

    return get_acl_dtype(elem.dtype)


def _get_tensor_info(node: Any) -> TensorInfo:
    dims, _ = get_shape(node)
    shape_str = _process_shape(tuple(dims))
    dtype = get_dtype(node)
    return TensorInfo(shape_str, str(get_torch_dtype(dtype)))


def _create_tensor_desc(
    shape_str: str, dtype_str: str, input_name: Optional[str] = None
) -> TensorDesc:
    return TensorDesc(dtype=dtype_str, shape=shape_str, input=input_name)


def make_output_tensor_desc(
    output_names: List[str],
    output_data_nodes: List[Any],
    input_data_nodes: List[Any],
    graph_outputs: List[str],
    inplace_tensor_dict: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    tensor_descs = {}
    input_node_map = {node.name: node for node in input_data_nodes}

    def process_output(output_name: str, node: Optional[Any] = None) -> None:
        if output_name in inplace_tensor_dict:
            input_name = inplace_tensor_dict[output_name]
            input_node = input_node_map.get(input_name)
            assert input_node is not None, f"Input node for {input_name} not found"
            tensor_info = _get_tensor_info(input_node)
            tensor_desc = _create_tensor_desc(*tensor_info, input_name)
        else:
            assert node is not None, f"Node for {output_name} not found"
            tensor_info = _get_tensor_info(node)
            tensor_desc = _create_tensor_desc(*tensor_info)

        tensor_descs[output_name] = dataclasses.asdict(tensor_desc)

    for output_name, output_node in zip(output_names, output_data_nodes):
        process_output(output_name, output_node)

    return tensor_descs


def _process_tuple_operations(graph: Graph) -> Dict[str, str]:
    tuple_replace = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "tupleOperation":
            for idx, input_name in enumerate(node.inputs):
                key_name = f"{node.op_name}__{idx}"
                tuple_replace[key_name] = input_name
            del graph.nodes[name]
    return tuple_replace


def _process_getitem_operations(
    graph: Graph, tuple_replace: Dict[str, str], output_names: List[str]
) -> Dict[str, str]:
    getitem_replace = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "getitemOperation":
            real_name = f"{node.inputs[0]}__{node.index}"
            real_name = tuple_replace.get(real_name, real_name)
            real_name = getitem_replace.get(real_name, real_name)
            getitem_replace[node.outputs[0]] = real_name
            del graph.nodes[name]

    for node in graph.nodes.values():
        if not isinstance(node, GraphOpearation):
            node.inputs = [getitem_replace.get(x, x) for x in node.inputs]
        else:
            node.outputs = [getitem_replace.get(x, x) for x in node.outputs]

    for idx, output in enumerate(output_names):
        output_names[idx] = getitem_replace.get(output, output)
    return getitem_replace


def _update_infer_shape(graph_node: GraphOpearation) -> None:
    infer_shape = []
    for item in graph_node.infer_shape["value"]:
        node_id, tensor_id = item
        node_name = graph_node.node_names[node_id]
        input_name = graph_node.nodes[node_name].inputs[tensor_id]
        infer_shape.append(graph_node.inputs.index(input_name))

    graph_node.infer_shape = {"type": "equal", "value": infer_shape}


def _handle_inplace_outputs(
    graph_node: GraphOpearation, inplace_output_tensors: Dict[str, str]
) -> None:
    graph_node.has_inplace_output = True
    for output_name, input_name in inplace_output_tensors.items():
        if output_name in graph_node.outputs and input_name in graph_node.inputs:
            output_idx = graph_node.outputs.index(output_name)
            input_idx = graph_node.inputs.index(input_name)
            graph_node.add_inplace_output(output_idx, input_idx)


def _process_graph_operations(graph: Graph) -> None:
    graph_nodes = [
        node for node in graph.nodes.values() if isinstance(node, GraphOpearation)
    ]

    for graph_node in graph_nodes:
        graph_inputs = OrderedDict()
        graph_outputs = OrderedDict()
        graph_hosts = OrderedDict()
        inplace_output_tensors = OrderedDict()

        for node_name in graph_node.node_names:
            if node_name not in graph.nodes:
                continue

            node = graph.nodes[node_name]
            graph_node.nodes[node_name] = node
            del graph.nodes[node_name]

            graph_inputs.update(OrderedDict.fromkeys(node.inputs))
            graph_outputs.update(OrderedDict.fromkeys(node.outputs))

            if node.has_host_inputs:
                graph_hosts.update(OrderedDict.fromkeys(node.host_inputs))

            if node.special_constants_map:
                graph.update_special_constants(node.special_constants_map)

            if node.has_inplace_output:
                for item in node.inplace_outputs:
                    output_idx = item["output_index"]
                    input_idx = item["input_index"]
                    output_name = node.outputs[output_idx]
                    input_name = node.inputs[input_idx]
                    inplace_output_tensors[output_name] = input_name

        filtered_inputs = OrderedDict.fromkeys(
            name for name in graph_inputs if name not in graph_outputs
        )

        filtered_internals = OrderedDict.fromkeys(
            name
            for name in graph_outputs
            if name not in filtered_inputs and name not in graph_node.outputs
        )

        graph_node.set_inputs(list(filtered_inputs.keys()))
        graph_node.set_internals(list(filtered_internals.keys()))
        graph_node.node_names = list(graph_node.nodes.keys())

        if graph_node.has_infer_shape:
            _update_infer_shape(graph_node)

        if graph_hosts:
            graph_node.has_host_inputs = True
            graph_node.host_inputs = list(graph_hosts.keys())

        if inplace_output_tensors:
            _handle_inplace_outputs(graph_node, inplace_output_tensors)


def _collect_and_update_tensor_info(graph: Graph) -> Tuple[List[str], List[str]]:
    all_tensors = OrderedDict()
    host_tensors = OrderedDict()

    for node in graph.nodes.values():
        all_tensors.update(OrderedDict.fromkeys(node.inputs))
        all_tensors.update(OrderedDict.fromkeys(node.outputs))

        if node.has_host_inputs:
            host_tensors.update(OrderedDict.fromkeys(node.host_inputs))

        if node.special_constants_map:
            graph.update_special_constants(node.special_constants_map)

    return list(all_tensors.keys()), list(host_tensors.keys())


def _process_node_io(
    graph: Graph, input_names: List[str], host_tensors: List[str]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    node_inputs_count = OrderedDict.fromkeys(input_names, 0)
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
    node_inputs = [tensor for tensor, count in node_inputs_count.items() if count > 0]

    return node_inputs, node_hosts


def _collect_graph_inplace_out(graph: Graph, input_names: List[str]) -> Dict[str, str]:
    inplace_output_info = {}

    for output in graph.outputs:
        output_node = None
        output_node_name = None

        for node_name, node in graph.nodes.items():
            if output in node.outputs:
                output_node = node
                output_node_name = node_name
                break

        if output_node is None or not output_node.has_inplace_output:
            continue

        graph_output_index = output_node.outputs.index(output)
        input_index = next(
            (
                item["input_index"]
                for item in output_node.inplace_outputs
                if item["output_index"] == graph_output_index
            ),
            -1,
        )

        if input_index == -1:
            continue

        node_input_name = output_node.inputs[input_index]

        if node_input_name not in input_names:
            continue

        if output_node.is_reshape_op:
            # Reshape operations are defaulted to be inplace.
            # However, when the operator's input is also an input to the graph,
            # and the operator's output is an output of the graph, it cannot be inplace.
            graph.nodes[output_node_name].has_inplace_output = False
        else:
            inplace_output_info[output] = node_input_name

    return inplace_output_info


def parse_graph(
    graph: Graph,
    input_names,
    output_names,
    input_data_nodes,
    output_data_nodes,
    py_output_names,
):
    tuple_replace = _process_tuple_operations(graph)
    getitem_replace = _process_getitem_operations(graph, tuple_replace, output_names)

    _process_graph_operations(graph)

    all_tensors, host_tensors = _collect_and_update_tensor_info(graph)

    node_inputs, node_hosts = _process_node_io(graph, input_names, host_tensors)
    node_outputs = output_names.copy()

    node_internals = [
        tensor
        for tensor in all_tensors
        if (
            tensor not in node_inputs
            and tensor not in node_outputs
            and tensor not in graph.special_constants_map
        )
    ]
    graph.set_inputs(node_inputs)
    graph.set_outputs(node_outputs)
    graph.set_internals(node_internals)
    graph.set_hosts(node_hosts)
    graph.extend_inputs(list(graph.special_constants_map.keys()))

    inplace_out_info = _collect_graph_inplace_out(graph, input_names)

    for name in graph.nodes.keys():
        graph.nodes[name] = graph.nodes[name].build()

    output_tensor_descs = make_output_tensor_desc(
        output_names,
        output_data_nodes,
        input_data_nodes,
        node_outputs,
        inplace_out_info,
    )

    py_output_names = [getitem_replace.get(x, x) for x in py_output_names]

    return graph, output_tensor_descs, py_output_names
