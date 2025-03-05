# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from functools import lru_cache
from dlinfer.vendor import vendor_name


vendor = ["camb", "ascend"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def vendor_device_init():
    import_vendor_module(vendor_name)


vendor_device_init()
