import importlib
import os, sys
import typing
from typing import Any, Dict, List, Optional, Union
import transformers
from .patch import apply_model_patches


def patched_get_class_in_module(
    class_name: str, module_path: Union[str, os.PathLike]
) -> typing.Type:
    ret_class = transformers_get_class_in_module(class_name, module_path)
    apply_model_patches(importlib.import_module(ret_class.__module__))
    return ret_class


transformers_get_class_in_module = transformers.dynamic_module_utils.get_class_in_module
transformers.dynamic_module_utils.get_class_in_module = patched_get_class_in_module
