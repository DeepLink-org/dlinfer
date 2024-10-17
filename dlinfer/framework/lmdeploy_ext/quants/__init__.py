# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from functools import lru_cache
from dlinfer.vendor import vendor_name


awq_vendor = ["ascend"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in awq_vendor:
        importlib.import_module(f".{vendor_name_str}_awq", __package__)


def vendor_quant_init():
    import_vendor_module(vendor_name)


vendor_quant_init()
