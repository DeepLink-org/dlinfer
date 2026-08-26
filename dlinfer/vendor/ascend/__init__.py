# Copyright (c) 2024, DeepLink. All rights reserved.
from .version import ensure_ascend_runtime

ensure_ascend_runtime()

from . import pytorch_patch, torch_npu_ops  # noqa: E402,F401
