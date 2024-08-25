// Copyright (c) 2024, DeepLink. All rights reserved.
#include "ascend_ops.hpp"
#include "op_api_common.hpp"
#include <ATen/ops/tensor.h>
#include <tuple>

namespace dlinfer {

namespace ascend {

::std::tuple<at::Tensor,at::Tensor> npu_moe_gating_topk_softmax(
    const at::Tensor &x, const at::Tensor &finishedOptional,
    int64_t topk, const at::Tensor y_out,
    const at::Tensor expert_idx_out, const at::Tensor row_idx_out)
{
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMoeGatingTopKSoftmax, x, finishedOptional, topk,
                                 y_out, expert_idx_out, row_idx_out);
    return std::tie(y_out, expert_idx_out);
}

} // namespace ascend

} // namespace dlinfer
