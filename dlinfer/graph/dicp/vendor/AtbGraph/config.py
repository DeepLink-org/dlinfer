import torch

from dlinfer.graph.dicp.dynamo_bridge.decompositions import (
    get_decompositions,
    register_decomposition_for_dicp,
)


aten = torch.ops.aten


@register_decomposition_for_dicp(aten.select.int)
def select_int(tensor, dim, index):
    if (
        not isinstance(tensor.shape[0], torch.SymInt)
        and tensor.shape[0] == 1
        and dim == 0
        and index == 0
    ):
        view_shape = [-1 if isinstance(x, torch.SymInt) else x for x in tensor.shape]
        del view_shape[0]
        return tensor.view(view_shape)
    slice_res = aten.slice.Tensor(tensor, dim, index, index + 1, 1)
    return slice_res.squeeze(dim)


def get_decomp():
    return get_decompositions(
        [
            aten.count_nonzero.default,
            aten.select.int,
        ]
    )


decomp = get_decomp()
