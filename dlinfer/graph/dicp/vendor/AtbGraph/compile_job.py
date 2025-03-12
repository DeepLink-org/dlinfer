import os
import subprocess
import time

import torch
from dlinfer.graph.dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import (
    pick_vec_isa,
    cpp_compile_command,
    write,
    code_hash,
)
from torch._inductor import exc

from dlinfer.graph.dicp.vendor.AtbGraph.codegen import load_and_run


class AtbCompileJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        picked_vec_isa = pick_vec_isa()
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._key, self._input_path = write(
            source_code.strip(),
            "json",
            # extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) +
            # 'local_rank' + str(self._local_rank)
            extra="local_rank" + str(self._local_rank),
        )

    def _compile(self):
        try:
            if not hasattr(torch.classes.DICPModel, "DICPModel"):
                if os.getenv("DICP_USE_TORCH_NPU_LAUNCHER", "1") == "1":
                    os.environ["ATB_CONTEXT_HOSTTILING_RING"] = "1"
                    os.environ["ATB_CONTEXT_HOSTTILING_SIZE"] = "102400"
                    os.environ["ATB_WORKSPACE_MEM_ALLOC_GLOBAL"] = "1"
                    os.environ["ATB_USE_TILING_COPY_STREAM"] = "0"
                    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT"] = "1"
                    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT"] = "16"
                current_dir = os.path.dirname(__file__)
                lib_path = os.path.join(current_dir, "codegen/libdicp_model.so")
                torch.classes.load_library(lib_path)
        except Exception as e:
            current_dir = os.path.dirname(__file__)
            lib_path = os.path.join(current_dir, "codegen/libdicp_model.so")
            torch.classes.load_library(lib_path)

    def get_key(self):
        return self._key

    def get_compile_result(self):
        self._compile()
        return load_and_run.AtbModel(self._input_path)
