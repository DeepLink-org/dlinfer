# Copyright (c) 2024, DeepLink. All rights reserved.
import os
import pathlib
from functools import lru_cache
from packaging import version

import torch
import torch_npu

origin_torch_compile = torch.compile
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils._path_manager import PathManager

torch.compile = origin_torch_compile

if version.parse(torch.__version__) >= version.parse("2.2.0"):
    from importlib import import_module

    target_module_str = "torch.utils._triton"
    target_module = import_module(target_module_str)
    func_str = "has_triton"

    def has_triton():
        return False

    setattr(target_module, func_str, has_triton)


@lru_cache(None)
def register_atb_extensions():
    npu_path = pathlib.Path(torch_npu.__file__).parents[0]
    atb_so_path = os.path.join(npu_path, "lib", "libop_plugin_atb.so")
    try:
        PathManager.check_directory_path_readable(atb_so_path)
        torch.ops.load_library(atb_so_path)
    except OSError as e:
        nnal_ex = None
        nnal_strerror = ""
        if "libatb.so" in str(e):
            nnal_strerror = (
                "Please check that the nnal package is installed. "
                "Please run 'source set_env.sh' in the NNAL installation path."
            )
        if "undefined symbol" in str(e):
            nnal_strerror = (
                "Please check the version of the NNAL package. "
                "An undefined symbol was found, "
                "which may be caused by a version mismatch between NNAL and torch_npu."
            )
        nnal_ex = OSError(e.errno, nnal_strerror)
        nnal_ex.__traceback__ = e.__traceback__
        raise nnal_ex from e


register_atb_extensions()
