import functools
import torch
from dlinfer.graph.dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)

atb_pattern_matcher = PatternMatcherPass()

torch_patterns_cls_list_1 = []
register_torch_pattern_1 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_1
)

torch_patterns_cls_list_2 = []
register_torch_pattern_2 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_2
)

torch_patterns_cls_list_3 = []
register_torch_pattern_3 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_3
)


aten = torch.ops.aten
atb = torch.ops.atb
dlinfer = torch.ops.dlinfer


@register_torch_pattern_1
class TorchLinear(BackendPatternBase):
    @staticmethod
    def pattern(x_input, weight, viewed_input_shape, viewed_output_shape):
        trans_weight = torch.ops.aten.t.default(weight)
        viewed_input = torch.ops.aten.view.default(x_input, viewed_input_shape)
        mm_result = torch.ops.aten.mm.default(viewed_input, trans_weight)
        viewed_mm_result = torch.ops.aten.view.default(mm_result, viewed_output_shape)
        return viewed_mm_result

    @staticmethod
    def replacement(x_input, weight):
        return torch.ops.atb.linear.default(x_input, weight, None, False, True)


@register_torch_pattern_1
class TorchLinearWithBias(BackendPatternBase):
    @staticmethod
    def pattern(bias, x_input, weight, viewed_input_shape, viewed_output_shape):
        trans_weight = torch.ops.aten.t.default(weight)
        viewed_input = torch.ops.aten.view.default(x_input, viewed_input_shape)
        addmm_result = torch.ops.aten.addmm.default(bias, viewed_input, trans_weight)
        viewed_mm_result = torch.ops.aten.view.default(
            addmm_result, viewed_output_shape
        )
        return viewed_mm_result

    @staticmethod
    def replacement(bias, x_input, weight):
        return torch.ops.atb.linear.default(x_input, weight, bias, False, True)


@register_torch_pattern_1
class TorchLInearAllreduce(BackendPatternBase):
    @staticmethod
    def pattern(x, weight, bias, allreduce, lmdeploy_group_type, group):
        linear = torch.ops.dlinfer.linear.default(
            x, weight, bias, allreduce, lmdeploy_group_type
        )
        all_reduce = torch.ops._c10d_functional.all_reduce.default(linear, "sum", group)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        copy = torch.ops.aten.copy.default(linear, wait_tensor)
        return copy

    @staticmethod
    def replacement(x, weight, bias, allreduce, lmdeploy_group_type, group):
        return torch.ops.dlinfer.linear.default(x, weight, bias, True, group)


@register_torch_pattern_1
class TorchAllreduce(BackendPatternBase):
    @staticmethod
    def pattern(x, group):
        all_reduce = torch.ops._c10d_functional.all_reduce.default(x, "sum", group)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        copy = torch.ops.aten.copy.default(x, wait_tensor)
        return copy

    @staticmethod
    def replacement(x, group):
        return torch.ops.atb.allreduce.default(x, "sum", group)


@register_torch_pattern_1
class TorchInplaceDivTensor(BackendPatternBase):
    @staticmethod
    def pattern(x, other):
        div = torch.ops.aten.div.Tensor(x, other)
        copy = torch.ops.aten.copy_.default(x, div)
        return copy

    @staticmethod
    def replacement(x, other):
        return torch.ops.atb.inplace_div.default(x, other)


@register_torch_pattern_1
class TorchInplaceScatterTensor(BackendPatternBase):
    @staticmethod
    def pattern(x, dim, index, src):
        scatter = torch.ops.aten.scatter.src(x, dim, index, src)
        copy = torch.ops.aten.copy_.default(x, scatter)
        return copy

    @staticmethod
    def replacement(x, dim, index, src):
        return torch.ops.atb.inplace_scatter.default(x, dim, index, src)
