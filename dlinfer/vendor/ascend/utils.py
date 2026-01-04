from functools import lru_cache
import torch
from dlinfer.utils.type_annotation import DlinferDistContext


class SocVersion:
    Ascend310P: str = "Ascend310P"
    Ascend910: str = "Ascend910"

    @classmethod
    @lru_cache(maxsize=1)
    def device_name(cls) -> str:
        return torch.npu.get_device_name()

    @classmethod
    def is_Ascend310P(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend310P)

    @classmethod
    def is_Ascend910(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend910)

    @classmethod
    @lru_cache(maxsize=1)
    def soc_version(cls) -> str:
        return torch.npu.get_soc_version()

    @classmethod
    def is_A2(cls) -> bool:
        return 220 <= cls.soc_version() <= 225

    @classmethod
    def is_A3(cls) -> bool:
        return 250 <= cls.soc_version() <= 255


@lru_cache(maxsize=1)
def get_cpu_seq_len(seq_len):
    return seq_len.cpu()


@lru_cache(maxsize=1)
def get_world_size_accros_dp(dist_ctx: DlinferDistContext) -> int:
    return dist_ctx.tp_size * dist_ctx.dp_size
