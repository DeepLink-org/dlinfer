"""Device correctness probe for DLInfer's bundled direct2560 extension."""

import argparse

from dlinfer.vendor.ascend import grouped_matmul_direct
import torch
import torch_npu


def run(dtype: torch.dtype, iterations: int = 1) -> None:
    if not grouped_matmul_direct.is_available():
        raise RuntimeError(grouped_matmul_direct.unavailable_reason())

    experts, hidden_size, output_size, tokens = 2560, 16, 16, 8
    device = torch.device("npu:0")
    x = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    stored_weight = torch.randn(
        experts, output_size, hidden_size, device=device, dtype=dtype
    )
    weight = stored_weight.transpose(1, 2)
    group_list = torch.zeros(experts, device=device, dtype=torch.int64)
    group_list[:tokens] = 1

    references = []
    row_start = 0
    for expert_start in range(0, experts, 1024):
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

    reference = torch.cat(references, dim=0)
    for iteration in range(iterations):
        print(f"ITERATION {iteration + 1}/{iterations} begin", flush=True)
        direct = grouped_matmul_direct.grouped_matmul(x, stored_weight, group_list, 1)
        print(f"ITERATION {iteration + 1}/{iterations} submitted", flush=True)
        torch_npu.npu.synchronize()
        print(f"ITERATION {iteration + 1}/{iterations} synchronized", flush=True)
        torch.testing.assert_close(direct, reference, rtol=0, atol=0)
    print(
        f"PASS dtype={dtype} iterations={iterations} "
        f"max_abs={(direct - reference).abs().max().item()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.dtype in ("fp16", "both"):
        run(torch.float16, args.iterations)
    if args.dtype in ("bf16", "both"):
        run(torch.bfloat16, args.iterations)


if __name__ == "__main__":
    main()
