# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from functools import lru_cache
from dlinfer.vendor import vendor_name


graph_vendor = ["maca", "camb"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in graph_vendor:
        importlib.import_module(f".{vendor_name_str}_cudagraph", __package__)


def vendor_graph_init():
    import_vendor_module(vendor_name)


vendor_graph_init()
