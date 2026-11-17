# Copyright (c) 2024, DeepLink. All rights reserved.
import os
from pathlib import Path


def _register_bundled_ascend_opp() -> None:
    """Expose wheel-local Ascend operators before torch_npu initializes."""
    if os.environ.get("DLINFER_GMM_EXPERIMENT", "chunked") != "direct2560":
        return
    ascend_dir = Path(__file__).parent / "vendor/ascend"
    candidates = (
        ascend_dir / "grouped_matmul_direct",
        ascend_dir / "csrc/grouped_matmul_direct/vendor",
    )
    bundled_opp = next((path for path in candidates if path.is_dir()), None)
    if bundled_opp is None:
        return
    bundled_opp_str = str(bundled_opp)
    configured = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    paths = [path for path in configured.split(":") if path]
    if bundled_opp_str not in paths:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([bundled_opp_str, *paths])


_register_bundled_ascend_opp()

import dlinfer.vendor as vendor

vendor.vendor_torch_init()
__version__ = "0.2.8"
