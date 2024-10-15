# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from pathlib import Path
from functools import lru_cache
import yaml


awq_vendor = ["ascend"]
vendor_name_file = Path(__file__).parent.parent.parent.parent / "vendor/vendor.yaml"
with open(str(vendor_name_file), "r") as f:
    config = yaml.safe_load(f)
    vendor_name = config["vendor"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in awq_vendor:
        importlib.import_module(f".{vendor_name_str}_awq", __package__)


def vendor_torch_init():
    import_vendor_module(vendor_name)


vendor_torch_init()
