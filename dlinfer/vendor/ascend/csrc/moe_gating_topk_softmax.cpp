#include <ATen/ops/tensor.h>

#include <tuple>

#include "ascend_ops.hpp"
#include "op_api_common.hpp"

namespace dlinfer {

namespace ascend {

::std::tuple<at::Tensor, at::Tensor> npu_moe_gating_topk_softmax(const at::Tensor& x, const c10::optional<at::Tensor>& finished_opt, int64_t topk,
                                                                 at::Tensor& y_out, at::Tensor& expert_idx_out, at::Tensor& row_idx_out) {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMoeGatingTopKSoftmax, x, finished_opt, topk, y_out, expert_idx_out, row_idx_out);
    return std::tie(y_out, expert_idx_out);
}

}  // namespace ascend

}  // namespace dlinfer
