import filecmp
import importlib
import importlib.util
import os
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import transformers
from transformers.utils import HF_MODULES_CACHE
from .patch_transformers.patch import apply_model_patches

def patched_get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_spec = importlib.util.spec_from_file_location(name, location=Path(HF_MODULES_CACHE) / module_path)
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(module_spec)
        # insert it into sys.modules before any loading begins
        sys.modules[name] = module
    # reload in both cases
    module_spec.loader.exec_module(module)
    apply_model_patches(module, name)
    return getattr(module, class_name)

transformers.dynamic_module_utils.get_class_in_module = patched_get_class_in_module
