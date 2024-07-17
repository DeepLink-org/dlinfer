import torch
import infer_ext

x = torch.randn(17, 8).cuda()
topk = 2

ext_res = infer_ext.vendor.ascend.moe_gating_topk_softmax(x, topk)

routing_weights = torch.softmax(x, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(routing_weights, topk, dim=-1)

print(torch.eq(routing_weights, ext_res[0]))
print(torch.eq(selected_experts, ext_res[1]))