# Copyright (c) 2024, DeepLink. All rights reserved.
from datetime import timedelta
import torch.distributed as dist
from torch.distributed import (
    Backend,
    Store,
    default_pg_timeout,
)

from dlinfer.utils.type_annotation import *
from dlinfer.vendor import get_comm_str


# change nccl to internal used dicl, so existing model can keep nccl as backend name.
_raw_init_process_group = dist.init_process_group


def _wrap_init_process_groups(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: timedelta = default_pg_timeout,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
):
    if backend == None or backend == Backend.NCCL:
        backend = get_comm_str()
    _raw_init_process_group(
        backend, init_method, timeout, world_size, rank, store, group_name, pg_options
    )

def apply_dist_patch():
    dist.init_process_group = _wrap_init_process_groups
