# Copyright (c) 2024, DeepLink. All rights reserved.
from dlinfer.vendor import vendor_name


def torch_npu_ext_init():
    """Initialize torch npu extensions."""
    if vendor_name == "ascend":
        from .aclgraph import appy_patch

        appy_patch()


torch_npu_ext_init()
