#include "ascend_ops.hpp"
#include "utils.hpp"

namespace infer_ext {

namespace ascend_ops {

at::Tensor moe_topk_gating_softmax(at::Tensor routing_weights, at::Tensor selected_experts,
                               at::Tensor selected_idx, at::Tensor router_logits, int64_t topk) {
    at::Tensor finishedOptional = at::Tensor();
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMoeGatingTopKSoftmax,
                                 router_logits,
                                 finishedOptional,
                                 topk,
                                 routing_weights,
                                 selected_experts,
                                 selected_idx);
    return router_logits
}

} // namespace ascend_ops

} // namespace infer_ext 
