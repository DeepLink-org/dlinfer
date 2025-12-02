from functools import lru_cache
import torch


class SocVersion:
    Ascend310P: str = "Ascend310P"
    Ascend910: str = "Ascend910"
    device_name: str = torch.npu.get_device_name()

    @classmethod
    def is_Ascend310P(cls) -> bool:
        return cls.device_name.startswith(cls.Ascend310P)

    @classmethod
    def is_Ascend910(cls) -> bool:
        return cls.device_name.startswith(cls.Ascend910)
