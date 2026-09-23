import pytest

from dlinfer.vendor.ascend import grouped_matmul_direct
import torch


torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.skipif(not torch_npu.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_matmul_direct2560_matches_chunked(dtype):
    assert (
        grouped_matmul_direct.is_available()
    ), grouped_matmul_direct.unavailable_reason()

    experts = 2560
    hidden_size = 16
    output_size = 16
    tokens = 8
    device = torch.device("npu:0")

    x = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    # Production stores weights as [E, N, K] and passes a transposed view.
    stored_weight = torch.randn(
        experts, output_size, hidden_size, device=device, dtype=dtype
    )
    weight = stored_weight.transpose(1, 2)
    group_list = torch.zeros(experts, device=device, dtype=torch.int64)
    group_list[:tokens] = 1

    direct = grouped_matmul_direct.grouped_matmul(x, stored_weight, group_list, 1)

    references = []
    row_start = 0
    expert_start = 0
    while expert_start < experts:
        expert_end = min(expert_start + 1024, experts)
        chunk_groups = group_list[expert_start:expert_end]
        row_end = row_start + int(chunk_groups.sum().cpu())
        if row_end > row_start:
            references.append(
                torch.ops.npu.npu_grouped_matmul(
                    [x[row_start:row_end]],
                    [weight[expert_start:expert_end]],
                    group_list=chunk_groups,
                    split_item=2,
                    group_type=0,
                    group_list_type=1,
                )[0]
            )
        row_start = row_end
        expert_start = expert_end

    reference = torch.cat(references, dim=0)
    torch.testing.assert_close(direct, reference, rtol=0, atol=0)
