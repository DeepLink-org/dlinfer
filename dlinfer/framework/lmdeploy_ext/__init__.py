# Copyright (c) 2024, DeepLink. All rights reserved.
import dlinfer.framework.transformers_ext
import importlib
from pathlib import Path
from functools import lru_cache
import yaml


vendor_is_initialized = False
vendor_name_file = Path(__file__).parent.parent.parent / "vendor/vendor.yaml"
with open(str(vendor_name_file), "r") as f:
    config = yaml.safe_load(f)
    vendor_name = config["vendor"]
    dispatch_key = config["dispatch_key"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    return importlib.import_module(f".{vendor_name_str}", __package__)


def vendor_torch_init():
    import_vendor_module(vendor_name)
    global vendor_is_initialized
    vendor_is_initialized = True


vendor_torch_init()
