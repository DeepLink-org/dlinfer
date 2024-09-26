// Copyright (c) 2024, DeepLink. All rights reserved.
#include <ATen/core/ATen_fwd.h>
#include <c10/util/OptionalArrayRef.h>

#include "ascend_ops.hpp"
#include "op_api_common.hpp"

namespace dlinfer {

namespace ascend {

at::Tensor npu_weight_quant_batchmatmul_out(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& antiquant_scale,
                                            const at::optional<at::Tensor>& antiquant_offset, const at::optional<at::Tensor>& quant_scale,
                                            const at::optional<at::Tensor>& quant_offset, const at::optional<at::Tensor>& bias, int64_t group_size,
                                            at::Tensor matmul_output) {
    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnWeightQuantBatchMatmulV2, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, group_size, matmul_output);
    return matmul_output;
}

}  // namespace ascend

}  // namespace dlinfer
