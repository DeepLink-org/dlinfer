from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any


@dataclass
class RopeParam:
    rotaryCoeff: int = 2
    cosFormat: int = 0

class SelfAttentionCalcType(IntEnum):
    UNDEFINED = 0
    ENCODER = 1
    DECODER = 2
    PA_ENCODER = 3

class SelfAttentionKernelType(IntEnum):
    KERNELTYPE_DEFAULT = 0
    KERNELTYPE_HIGH_PRECISION = 1

class SelfAttentionClampType(IntEnum):
    CLAMP_TYPE_UNDEFINED = 0
    CLAMP_TYPE_MIN_MAX = 1

class SelfAttentionMaskType(IntEnum):
    MASK_TYPE_UNDEFINED = 0
    MASK_TYPE_NORM = 1
    MASK_TYPE_ALIBI = 2
    MASK_TYPE_NORM_COMPRESS = 3
    MASK_TYPE_ALIBI_COMPRESS = 4
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 6

@dataclass
class SelfAttentionParam:
    headNum: int = 0
    kvHeadNum: int = 0
    qScale: float = 1.0
    qkScale: float = 1.0
    batchRunStatusEnable: bool = False
    isTriuMask: int = 0
    calcType: SelfAttentionCalcType = SelfAttentionCalcType.UNDEFINED
    kernelType: SelfAttentionKernelType = SelfAttentionKernelType.KERNELTYPE_DEFAULT
    clampType: SelfAttentionClampType = SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
    clampMin: float = 0.0
    clampMax: float = 0.0
    maskType: SelfAttentionMaskType = SelfAttentionMaskType.MASK_TYPE_UNDEFINED


class PagedAttentionMaskType(IntEnum):
    UNDEFINED = 0
    MASK_TYPE_NORM = 1
    MASK_TYPE_ALIBI = 2
    MASK_TYPE_SPEC = 3

class PagedAttentionQuantType(IntEnum):
    TYPE_QUANT_UNDEFINED = 0
    TYPE_DEQUANT_FUSION = 1

class PagedAttentionCompressType(IntEnum):
    COMPRESS_TYPE_UNDEFINED = 0
    COMPRESS_TYPE_KVHEAD = 1

class PagedAttentionCalcType(IntEnum):
    CALC_TYPE_UNDEFINED = 0
    CALC_TYPE_SPEC = 1

@dataclass
class PagedAttentionParam:
    headNum: int = 0
    qkScale: float = 1.0
    kvHeadNum: int = 0
    maskType: PagedAttentionMaskType = PagedAttentionMaskType.UNDEFINED
    batchRunStatusEnable: bool = False
    quantType: PagedAttentionQuantType = PagedAttentionQuantType.TYPE_QUANT_UNDEFINED
    hasQuantOffset: bool = False
    compressType: PagedAttentionCompressType = PagedAttentionCompressType.COMPRESS_TYPE_UNDEFINED
    calcType: PagedAttentionCalcType = PagedAttentionCalcType.CALC_TYPE_UNDEFINED

def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, IntEnum):
            return obj.value
        return obj
    return {k: convert_value(v) for k, v in data}

def to_dict(data):
    return asdict(data, dict_factory=custom_asdict_factory)
