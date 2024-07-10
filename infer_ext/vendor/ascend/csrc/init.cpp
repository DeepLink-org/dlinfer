#include "ascend_ops.hpp"
#include <torch/library.h>

using namespace infer_ext::ascend_ops;

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("moe_topk_gating_softmax", TORCH_FN(moe_topk_gating_softmax));
}
