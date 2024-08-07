# import torch
# import infer_ext.ops as ext_ops
# import pytest

# class OpModule(torch.nn.Module):
#     def forward(self, x, topk, topk_dim=-1):
#         routing_weights = torch.softmax(x, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, topk, dim=topk_dim)
#         return routing_weights, selected_experts
    
# model = OpModule()


# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("shape", [(17, 8), (15, 9)])
# @pytest.mark.parametrize("topk", [2, 3, 4])
# def test_moe_gating_softmax(dtype, shape, topk):
#     x = torch.randn(shape, dtype=dtype).cuda()
#     # ext_output = infer_ext.vendor.ascend.moe_gating_topk_softmax(x, topk)
#     ext_output = ext_ops.moe_gating_topk_softmax(x, topk)
#     torch_output = model(x, topk)
#     for i, item in enumerate(ext_output):
#         if isinstance(item, torch.Tensor):
#             assert torch.allclose(item.cpu().to(torch_output[i].dtype), torch_output[i].cpu(), equal_nan=True)
#         else:
#             assert item == torch_output[i]