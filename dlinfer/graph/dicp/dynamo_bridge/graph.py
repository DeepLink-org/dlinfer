import logging
import os
import torch
import torch.fx
from typing import List, Optional, Tuple
from torch._dynamo.utils import dynamo_timed
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._inductor.codecache import cache_dir
from dlinfer.graph.dicp.dynamo_bridge.utils import save_cpu_gm
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

log = logging.getLogger(__name__)


class GraphTransformer:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend: str,
    ):
        self.gm = gm
        self.backend = backend
        self.folder = cache_dir()
        self.cpu_gm, self.graph_key = save_cpu_gm(gm, self.folder)
        if backend == "topsgraph":
            from dlinfer.graph.dicp.vendor.TopsGraph.opset_transform import (
                topsgraph_opset_transform,
            )

            self.backend_opset_transform = topsgraph_opset_transform
            from dlinfer.graph.dicp.vendor.TopsGraph.codegen.enflame import (
                EnflameCodegen,
            )

            self.backend_codegen = EnflameCodegen
        elif backend == "ascendgraph":
            from dlinfer.graph.dicp.vendor.AscendGraph.opset_convert import (
                ascendgraph_opset_convert,
            )

            self.backend_opset_transform = ascendgraph_opset_convert
            from dlinfer.graph.dicp.vendor.AscendGraph.codegen.ascend import (
                AscendCodegen,
            )

            self.backend_codegen = AscendCodegen
        elif backend == "atbgraph":
            from dlinfer.graph.dicp.vendor.AtbGraph.opset_convert import (
                atbgraph_opset_convert,
            )

            self.backend_opset_transform = atbgraph_opset_convert
            from dlinfer.graph.dicp.vendor.AtbGraph.codegen.atb import AtbCodegen

            self.backend_codegen = AtbCodegen

    def transform(self):
        self.gm = self.backend_opset_transform(self.gm)

    @staticmethod
    def infer_shape_dtype(gm: torch.fx.GraphModule):
        def make_tensor_meta(x) -> Optional[TensorMetadata]:
            if isinstance(x, FakeTensor):
                return _extract_tensor_metadata(x)
            else:
                return None

        test_infer = bool(os.environ.get("TEST_DICP_INFER", False))
        for n in gm.graph.nodes:
            fake_value = None
            if n.op == "call_function":
                try:
                    fake_value = n.target(*n.args, **n.kwargs)
                except Exception as e:
                    raise RuntimeError(f"call function: {n.target} failed!")
            elif n.op == "get_attr":
                target_atoms = n.target.split(".")
                attr_itr = gm
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(
                            f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                        )
                    attr_itr = getattr(attr_itr, atom)
                    attr_size, attr_dtye = attr_itr.shape, attr_itr.dtype
                with FakeTensorMode():
                    fake_value = torch.empty(attr_size, dtype=attr_dtye)
            else:
                continue
            if "val" in n.meta and test_infer:
                (n_meta_val, fake_val) = (
                    ((n.meta["val"],), (fake_value,))
                    if not isinstance(n.meta["val"], (Tuple, List))
                    else (n.meta["val"], fake_value)
                )
                for i, (meta_i, fv_i) in enumerate(zip(n_meta_val, fake_val)):
                    if not isinstance(fv_i, FakeTensor):
                        continue
                    log_info = f"target: {n.target}, meta_i: {meta_i}, fv_i: {fv_i}"
                    assert (
                        meta_i.size() == fv_i.size()
                    ), f"check infer size failed, {log_info}"
                    assert (
                        meta_i.dtype == fv_i.dtype
                    ), f"check infer dtype failed, {log_info}"
                    assert (
                        meta_i.stride() == fv_i.stride()
                    ), f"check infer stride failed, {log_info}"
                    assert (
                        meta_i.storage_offset() == fv_i.storage_offset()
                    ), f"check infer storage offset failed, {log_info}"
            if "val" not in n.meta:
                n.meta["val"] = fake_value
                n.meta["tensor_meta"] = make_tensor_meta(n.meta["val"])
            # Modify strides of channels last tensor to keep contiguous.
            if os.getenv("DICP_SD_CLAST", default="False") == "True":
                n.meta["val"] = (
                    fake_value.contiguous()
                    if isinstance(fake_value, FakeTensor)
                    else fake_value
                )
                n.meta["tensor_meta"] = make_tensor_meta(n.meta["val"])

    def codegen(self):
        return self.backend_codegen(
            self.gm, self.cpu_gm, self.folder, self.graph_key
        ).codegen()

    @dynamo_timed
    def compile_to_module(self):
        from torch._inductor.codecache import PyCodeCache

        code = self.codegen()

        mod = PyCodeCache.load(code)

        # if dynamo_config.output_code:
        #     log.info("Output code: %s", mod.__file__)
        return mod

    def compile_to_fn(self):
        return self.compile_to_module().call
