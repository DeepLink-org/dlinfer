import importlib
import os, sys, re
import typing
from typing import Any, Dict, List, Optional, Union
import transformers
from .patch import apply_model_patches

def patched_get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    ret_class = transformers_get_class_in_module(class_name, module_path)
    apply_model_patches(importlib.import_module(ret_class.__module__))
    return ret_class

def patched_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Extracts all the libraries (not relative imports this time) that are imported in a file.

    Args:
        filename (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of all packages required to use the input module.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # filter out try/except block so in custom code we can have try/except imports
    content = re.sub(r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL)

    # Imports of the form `import xxx`
    imports = re.findall(r"^\s*import\s(\S)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports = re.findall(r"^\s*from\s(\S)\simport", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = set([imp.split(".")[0] for imp in imports if not imp.startswith(".")])
    imports.remove("flash_attn") if "flash_attn" in imports else ...
    return list(imports)


transformers_get_class_in_module = transformers.dynamic_module_utils.get_class_in_module
transformers.dynamic_module_utils.get_class_in_module = patched_get_class_in_module
transformers.dynamic_module_utils.get_imports = patched_get_imports
