#include <torch/torch.h>

enum class Status {
    Success = 0,
    Failed = -1,
};

namespace infer_ext {

namespace ascend_ops {

Status moe_topk_gating_softmax(
    at::Tensor routing_weights,
    at::Tensor selected_experts,
    at::Tensor selected_idx,
    at::Tensor router_logits,
    int64_t topk
);

} // namespace ascend_ops

} // namespace infer_ext 
