import os
import torch
from typing import Any, List
from torch.fx.node import Node

from torch._inductor.utils import IndentedBuffer
from dlinfer.graph.dicp.dynamo_bridge.utils import symint_in_shape
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.atb_graph import Graph, parse_graph
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.atb_op import AtbOverrides

graph_id = 0


def get_graph_id():
    global graph_id
    graph_id = graph_id + 1
    return graph_id


def process_name(name, target):
    if hasattr(target, "name"):
        real_op = target.name().split("::")[-1]
        if real_op.find(".") != -1:
            real_op = real_op.split(".")[0]
    else:
        real_op = name.rsplit("_", 1)[0] if name[-1].isdigit() else name
    return real_op


class AtbCodegen(torch.fx.Interpreter):
    def __init__(self, graph, aten_graph=None, folder=None, graph_key=None):
        self.graph = graph
        self.aten_graph = aten_graph
        self.override = AtbOverrides

        self.import_code = IndentedBuffer()
        self.build_graph_code = IndentedBuffer(initial_indent=1)

        self.graph_id = str(get_graph_id())
        self.args_dict = {}
        self.input_args = []
        self.output_args = []

        self.graph_input_names = []
        self.py_output_names = []
        self.graph_output_names = []

        self.sym_to_inputs = {}
        self.sym_in_args = {}

        self.atb_graph = Graph(str(get_graph_id()))

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name
        self.input_args.append(self.cur_node)

        fake_tensor = self.cur_node.meta["val"]
        if isinstance(fake_tensor, torch.SymInt):
            self.sym_to_inputs[fake_tensor.node.str()] = name
        elif symint_in_shape(fake_tensor.shape):
            # mention symint position in args
            # dynamic shape feature
            for idx, dim in enumerate(fake_tensor.shape):
                if isinstance(dim, torch.SymInt):
                    st = dim.node.str()
                    if st not in self.sym_in_args:
                        self.sym_in_args[st] = (name, idx)
        self.graph_input_names.append(self.args_dict[name])

    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            self.args_dict[name] = name

        _, args_list = AtbOverrides.gen_args(self.args_dict[name], self.args_dict, args)
        real_op = process_name(name, target)
        op = getattr(self.override, real_op)(*args_list, **kwargs)
        self.atb_graph.add_node(op)

    def call_method(self, name, target, args, kwargs):
        pass

    def output(self, name, target, args, kwargs):
        for arg in args:
            self.output_args.extend(arg)

    def run_node(self, n: Node) -> Any:
        self.cur_node = n
        op = n.op
        name = n.name
        target = n.target
        args = n.args
        kwargs = n.kwargs

        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        return getattr(self, op)(name, target, args, kwargs)

    def codegen(self):
        self.run()
        return self.generate_code()

    def parse_outputs(self):
        symint_inputs = self.sym_to_inputs.values()
        real_output_args = []
        for node in self.output_args:
            if isinstance(node, torch.fx.node.Node):
                name = self.args_dict[node.name]
                self.py_output_names.append(name)
                if name in self.graph_output_names or name in self.graph_input_names:
                    continue
                else:
                    real_output_args.append(node)
                    self.graph_output_names.append(name)
            else:
                self.py_output_names.append(str(node))
        self.output_args = real_output_args

    def gen_import_code(self):
        self.import_code.splice(
            """
                import os
                import torch
                import torch_npu
                import random
                import json
                from torch import empty_strided, as_strided, device
                from dlinfer.graph.dicp.dynamo_bridge.compile import AsyncCompileKernel
                from dlinfer.graph.dicp.vendor.AtbGraph.compile_job import AtbCompileJob
                
                # print('### codegen python file path: ', os.path.abspath(__file__))

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride

                def check_tensor(a, b, atol=5e-2, rtol=1e-2):
                    if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                        import pdb;pdb.set_trace()
                        pass
            """,
            strip=True,
        )
        return self.import_code.getvalue()

    def operator_in_str(self, st):
        for op in ["+", "-", "*", "/"]:
            if op in st:
                return True
        return False

    def gen_call_func(self):
        # TODO check scalar input
        call_body = IndentedBuffer()
        self.args = [self.args_dict[x.name] for x in self.input_args]
        call_body.writeline("""symInputs = []""")
        if len(self.args) == 1:
            call_body.writeline(f"{self.args[0]} = args[0]")
        else:
            call_body.writeline(f"({','.join(self.args)}) = args")

        # assign SymInt to InputArgs relationship
        if len(self.sym_in_args) > 0:
            for key in self.sym_in_args.keys():
                if not key.isdigit() and not self.operator_in_str(key):
                    call_body.writeline(
                        f"{key} = {self.sym_in_args[key][0]}.shape[{self.sym_in_args[key][1]}]"
                    )
        if len(self.sym_to_inputs) > 0:
            for key in self.sym_to_inputs.keys():
                if not key.isdigit() and not self.operator_in_str(key):
                    value = self.sym_to_inputs[key]
                    call_body.writeline(f"{key} = {value}")
                    call_body.writeline(
                        f"""symInputs.append('{{ "name": "{value}", "value": ' + str({key}) + ' }}')"""
                    )
                    call_body.writeline(
                        f"""symInputs.append('{{ "name": "{key}", "value": ' + str({key}) + ' }}')"""
                    )

        # gen fixed output shape
        graph_input_names = self.atb_graph.inputs
        graph_output_names = self.atb_graph.outputs

        for output in graph_output_names:
            create_info = self.output_tensor_descs["create"][output]
            if create_info["input"] is None:
                device = "npu"
                dtype = create_info["dtype"]
                shape = create_info["shape"]
                call_body.writeline(
                    f"""{output} = torch.empty({shape}, dtype={dtype}, device='{device}')"""
                )
            elif create_info["need_reshape"]:
                shape = create_info["shape"]
                input = create_info["input"]
                call_body.writeline(f"""{output} = {input}.view({shape})""")
            else:
                input = create_info["input"]
                call_body.writeline(f"""{output} = {input}""")

        call_body.writeline("""hostTensors = []""")
        call_body.writeline(f"""host_tensor_dict = {{}}""")
        host_tensors = []
        for tensor in self.atb_graph.hosts:
            node_id = tensor["nodeId"]
            tensor_id = tensor["tensorId"]
            tensor_name = tensor["tensorName"]
            assert tensor_name in self.args
            if tensor_name not in host_tensors:
                call_body.writeline(
                    f"""host_tensor_dict["{tensor_name}"] = {tensor_name}.cpu().tolist()"""
                )
                host_tensors.append(tensor_name)
                call_body.writeline(
                    f"""host_tensor_str_{tensor_name} = str(host_tensor_dict["{tensor_name}"])"""
                )
            call_body.writeline(
                f"""hostTensors.append('{{"nodeId": {node_id}, "tensorId": {tensor_id}, "value": ' + str(host_tensor_str_{tensor_name}) + ' }}')"""
            )

        call_body.writeline(
            """param = f'{{ \"symInputs\": [{",".join(symInputs)}], \"hostTensors\": [{",".join(hostTensors)}] }}'"""
        )
        call_body.writeline(f"""inputs = [{','.join(graph_input_names)}]""")

        call_body.writeline(f"""outputs = [{','.join(graph_output_names)}]""")
        call_body.writeline("kernel_cpp_0(inputs, outputs, param)")

        del_args = [f"del {x}" for x in self.args if x not in self.py_output_names]
        call_body.writelines(del_args)
        call_body.writeline("args.clear()")
        call_body.writeline(f"return ({', '.join(self.py_output_names)})")

        call_func = IndentedBuffer()
        call_func.writeline("def call(args):")
        with call_func.indent():
            call_func.splice(call_body)

        return call_func.getvalue()

    def gen_main_func(self):
        main_body = IndentedBuffer()
        main_body.splice(
            """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
            """,
            strip=True,
        )

        py_rand_inputs = []
        for i in range(len(self.input_args)):
            node = self.input_args[i]
            name = self.args[i]
            val = node.meta["val"]
            if isinstance(val, torch.SymInt):
                code_str = f"""{name} = random.randint(0, 4)"""
            else:
                shape = str(tuple(val.size()))
                stride = str(tuple(val.stride()))
                device = val.device.type
                dtype = str(val.dtype)
                code_str = f"""{name} = rand_strided({shape}, {stride}, device='{device}', dtype={dtype})"""
            py_rand_inputs.append(code_str)
        main_body.writelines(py_rand_inputs)
        main_body.writeline(
            f"print_performance(lambda: call([{', '.join(self.args)}]))"
        )

        main_func = IndentedBuffer()
        main_func.writeline("""if __name__ == "__main__":""")
        with main_func.indent():
            main_func.splice(main_body)
        return main_func.getvalue()

    def gen_graph_json(self):
        return self.atb_graph.to_json()

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        graph_json = self.gen_graph_json()
        compile_graph_code.splice(
            f"""
                atb_compile_job = AtbCompileJob('''{graph_json}''')
                async_compile = AsyncCompileKernel()
                kernel_cpp_0 = async_compile.compile_kernel(atb_compile_job)
            """,
            strip=True,
        )
        compile_graph_code.writeline("async_compile.wait(globals())")
        compile_graph_code.writeline("del async_compile")

        # special constant tensor for compiled graph
        if "rope_seqlen_default" in graph_json:
            compile_graph_code.writeline("\n")
            for name, expr in self.atb_graph.special_constants_map.items():
                compile_graph_code.writeline(f"{name} = {expr}")
        return compile_graph_code.getvalue()

    def generate_code(self):
        self.parse_outputs()
        self.atb_graph, self.output_tensor_descs, self.py_output_names = parse_graph(
            self.atb_graph,
            self.graph_input_names,
            self.graph_output_names,
            self.input_args,
            self.output_args,
            self.py_output_names,
        )
        return (
            self.gen_import_code()
            + self.gen_compile_graph_code()
            + self.gen_call_func()
            + self.gen_main_func()
        )
