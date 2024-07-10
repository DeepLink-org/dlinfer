#include "ascend_ops.hpp"
#include <torch/library.h>

using namespace infer_ext::ascend_ops;

namespace {

TORCH_LIBRARY(npu, m) {
    m.def("moe_topk_gating_softmax(Tensor routing_weights, Tensor selected_experts, Tensor selected_idx, Tensor router_logits, int) -> Tensor");
}

} // namespace

namespace {

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("moe_topk_gating_softmax", TORCH_FN(moe_topk_gating_softmax));
}

} // namespace
