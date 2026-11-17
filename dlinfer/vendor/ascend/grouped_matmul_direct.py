"""Bundled Ascend GroupedMatmul direct2560 extension."""

from typing import Optional

import torch
import torch_npu

_EXTENSION_IMPORT_ERROR: Optional[BaseException] = None

try:
    from . import _grouped_matmul_direct
except (ImportError, OSError) as exc:
    _grouped_matmul_direct = None
    _EXTENSION_IMPORT_ERROR = exc


def is_available() -> bool:
    """Whether the bundled pybind extension and op-api loaded successfully."""
    return _grouped_matmul_direct is not None


def unavailable_reason() -> str:
    if _EXTENSION_IMPORT_ERROR is None:
        return ""
    return str(_EXTENSION_IMPORT_ERROR)


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
) -> torch.Tensor:
    if _grouped_matmul_direct is None:
        raise RuntimeError(
            "DLInfer bundled GroupedMatmul extension is unavailable: "
            f"{unavailable_reason()}"
        )
    stream_handle = torch_npu.npu.current_stream(x.device).npu_stream
    return _grouped_matmul_direct.grouped_matmul(
        x, weight, group_list, group_list_type, stream_handle
    )
