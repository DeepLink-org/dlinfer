#pragma once

#include <atb/types.h>
#include <torch/torch.h>

#include <vector>

namespace dicp {
namespace tensor_utils {

std::string TensorToString(const atb::Tensor& tensor);
std::string TensorDescToString(const atb::TensorDesc& tensorDesc);

atb::Tensor AtTensor2Tensor(const at::Tensor& atTensor);
at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc& tensorDesc);
int64_t TransferAtTensor2AtbTensor(std::vector<torch::Tensor>& atTensors, std::vector<atb::Tensor>& atbTensors);

}  // namespace tensor_utils
}  // namespace dicp
