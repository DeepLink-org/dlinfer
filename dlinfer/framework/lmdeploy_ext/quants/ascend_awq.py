# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
import torch
from torch import nn

from typing import Optional, Type, TypeVar, Any, List
from lmdeploy.lite.quantization.modules.linear import WeightOnlyQLinear
from lmdeploy.lite.utils.cal_qparams import QParams
from lmdeploy.pytorch.distributed import get_world_rank
from lmdeploy.pytorch.nn.linear import (
    _chunk_align,
    MergedAwqLinear,
    AwqLinear,
    QKVAwqLinear,
)


def AscendWeightOnlyQLinear__init__(
    self,
    in_features: int,
    out_features: int,
    bias: Optional[torch.Tensor] = True,
    w_bit: int = 4,
    symmetry: bool = False,
    group_size: int = 128,
) -> None:
    super(WeightOnlyQLinear, self).__init__()

    assert w_bit == 4, "Only 4 bit are supported for ascend now."

    self.in_features = in_features
    self.out_features = out_features
    self.w_bit = w_bit
    self.group_size = group_size if group_size != -1 else in_features

    assert self.in_features % self.group_size == 0
    assert out_features % (32 // self.w_bit) == 0

    w_pack_oc = out_features // (32 // self.w_bit)
    w_inc = in_features
    weight = torch.zeros((w_inc, w_pack_oc), dtype=torch.int32)
    self.register_buffer("qweight", weight)

    if bias:
        self.register_buffer("bias", torch.zeros(out_features))
    else:
        self.bias = None

    s_inc = in_features // self.group_size
    s_oc = out_features
    scales = torch.zeros((s_inc, s_oc), dtype=torch.float16)
    self.register_buffer("scales", scales)

    if not symmetry:
        z_inc = in_features // self.group_size
        z_oc = out_features // 32
        zeros = torch.zeros((z_inc, z_oc), dtype=torch.float16)
        self.register_buffer("qzeros", zeros)
    else:
        self.qzeros = None


def AscendWeightOnlyQLinear_from_linear(
    cls: Type["WeightOnlyQLinear"],
    linear: nn.Linear,
    quantizer: TypeVar("Quantizer"),
    awq_layout: bool = True,
    qparams: Optional[QParams] = None,
) -> "WeightOnlyQLinear":
    """Create a WeightOnlyQLinear object from a PyTorch Linear object.

    Args:
        linear (nn.Linear): PyTorch Linear object.
        quantizer (Quantizer): Object that handles quantization.
        awq_layout (bool): AWQ layout. Defaults to True.

    Returns:
        WeightOnlyQLinear: A WeightOnlyQLinear object.
    """
    device = linear.weight.device

    w_bit = quantizer.bits
    pack_num = 32 // w_bit
    if awq_layout:
        assert w_bit == 4
        weight_pack_order = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("ascend only support awq")
    group_size = quantizer.group_size
    symmetry = quantizer.symmetry

    in_features = linear.in_features
    out_features = linear.out_features
    bias = False if linear.bias is None else True

    qlinear = cls(in_features, out_features, bias, w_bit, symmetry, group_size)
    qlinear.bias = linear.bias

    if qparams is None:
        qparams = quantizer.calculate_qparams(linear.weight)
        i32_w = quantizer.quant(linear.weight, qparams, real=True)
    else:
        i32_w = linear.weight.to(torch.int32)
    i32_w = i32_w.t().contiguous()

    pack_int_w = torch.zeros_like(qlinear.qweight).to(device)

    for col in range(pack_int_w.shape[1]):
        for i in range(pack_num):
            pack_int_w_col = i32_w[:, col * pack_num + weight_pack_order[i]]
            pack_int_w[:, col] |= pack_int_w_col << (i * w_bit)

    qlinear.qweight = torch.bitwise_xor(pack_int_w, 0x88888888)
    qlinear.scales = qparams.scales.squeeze(-1).t().contiguous()

    if qparams.zero_points is not None:
        zeros = qparams.zero_points.to(torch.int32).to(device)
        zeros = zeros.squeeze(-1).t().contiguous()
        qlinear.qzeros = 8.0 - zeros.to(torch.float16)
    qlinear.to("cpu")

    return qlinear


def AscendMergedAwqLinear__init__(
    self,
    in_features: int,
    all_out_features: List[int],
    w_bit: int,
    group_size: int,
    bias: bool,
    replicate: Optional[List[bool]] = None,
    device: Optional[torch.device] = None,
    is_tp: bool = True,
    out_names: Optional[List[int]] = None,
):
    if replicate is None:
        replicate = tuple(False for _ in all_out_features)

    self.split_section_s = all_out_features
    elem_per_int = 32 // w_bit
    self.split_section_wz = [size // elem_per_int for size in all_out_features]

    all_out_features = self._update_all_out_features(
        all_out_features, w_bit, group_size
    )
    self.all_out_features = all_out_features
    self.replicate = replicate
    if out_names is None:
        out_names = torch.arange(len(self.all_out_features)).tolist()
    assert len(out_names) == len(self.all_out_features)
    self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
    out_features = sum(all_out_features)
    super(MergedAwqLinear, self).__init__(
        in_features,
        out_features,
        w_bit,
        group_size,
        bias,
        device,
        colwise=True,
        is_tp=is_tp,
    )
    self.qweight.weight_loader = self.weight_loader
    self.qweight.weight_spliter = self.weight_spliter_wz
    self.qweight._weight_type = "qweight"
    self.scales.weight_loader = self.weight_loader
    self.scales.weight_spliter = self.weight_spliter_s
    self.scales._weight_type = "scales"
    self.qzeros.weight_loader = self.weight_loader
    self.qzeros.weight_spliter = self.weight_spliter_s
    self.qzeros._weight_type = "qzeros"
    if self.bias is not None:
        self.bias.weight_loader = self.weight_loader
        self.bias.weight_spliter = self.weight_spliter_s
        self.bias._weight_type = "bias"


def AscendMergedAwqLinear_weight_loader(
    self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any
):
    """weight loader."""
    world_size, rank = get_world_rank()
    shard_idx = self.out_names_map[shard_id]

    if loaded_weight.dim() == 1:
        # bias
        align = max(self.elem_per_int, self.group_size)
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        if not self.replicate[shard_idx]:
            weight = _chunk_align(loaded_weight, world_size, 0, align)[rank]
        param_w.copy_(weight)

    if param._weight_type in ["scales", "bias"]:
        # scales
        align = max(self.elem_per_int, self.group_size)
        param_w = param.data.split(self.all_out_features, -1)[shard_idx]
    elif param._weight_type in ["qzeros"]:
        align = max(self.elem_per_int, self.group_size)
        param_w = param.data.split(self.all_out_features, -1)[shard_idx]
    else:
        # qweight
        align = max(self.elem_per_int, self.group_size) // self.elem_per_int
        quanted_out_feats = [
            feat // self.elem_per_int for feat in self.all_out_features
        ]
        param_w = param.data.split(quanted_out_feats, 1)[shard_idx]

    if not self.replicate[shard_idx]:
        weight = _chunk_align(loaded_weight, world_size, -1, align)[rank]
    param_w.copy_(weight)


def AscendAwqLinear_create_weights(
    self,
    in_features: int,
    out_features: int,
    w_bit: int,
    group_size: int,
    bias: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    """create weights."""
    assert in_features % group_size == 0
    elem_per_int = 32 // w_bit
    assert out_features % elem_per_int == 0

    grouped_in_feats = in_features // group_size
    quant_out_feats = out_features // elem_per_int
    qweight = torch.empty(
        (in_features, quant_out_feats), dtype=torch.int32, device=device
    )
    scales = torch.empty((grouped_in_feats, out_features), dtype=dtype, device=device)
    qzeros = torch.empty((grouped_in_feats, out_features), dtype=dtype, device=device)
    if bias:
        bias = torch.empty((out_features,), dtype=dtype, device=device)
    else:
        bias = None
    return qweight, scales, qzeros, bias


WeightOnlyQLinear.__init__ = AscendWeightOnlyQLinear__init__
WeightOnlyQLinear.from_linear = classmethod(AscendWeightOnlyQLinear_from_linear)
MergedAwqLinear.__init__ = AscendMergedAwqLinear__init__
MergedAwqLinear.weight_loader = AscendMergedAwqLinear_weight_loader
AwqLinear.create_weights = AscendAwqLinear_create_weights
QKVAwqLinear.weight_loader = AscendMergedAwqLinear_weight_loader
