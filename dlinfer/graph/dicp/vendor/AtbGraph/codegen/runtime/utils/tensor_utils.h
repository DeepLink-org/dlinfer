#pragma once

#include <atb/types.h>
#include <torch/torch.h>

#include <vector>

namespace dicp {
namespace tensor_utils {

template <aclDataType T>
struct aclDataTypeMap;

template <>
struct aclDataTypeMap<aclDataType::ACL_FLOAT16> {
    using type = float16_t;
};
template <>
struct aclDataTypeMap<aclDataType::ACL_INT64> {
    using type = int64_t;
};
template <>
struct aclDataTypeMap<aclDataType::ACL_INT32> {
    using type = int32_t;
};
template <>
struct aclDataTypeMap<aclDataType::ACL_INT8> {
    using type = int8_t;
};

std::string TensorToString(const atb::Tensor& tensor);
std::string TensorDescToString(const atb::TensorDesc& tensorDesc);

atb::Tensor AtTensor2Tensor(const at::Tensor& atTensor);
at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc& tensorDesc);
int64_t TransferAtTensor2AtbTensor(std::vector<torch::Tensor>& atTensors, std::vector<atb::Tensor>& atbTensors);

template <aclDataType T>
void copyAndPrint(const atb::Tensor tensor, int64_t tensorSize);
int64_t DumpTensor(const atb::Tensor& tensor);

}  // namespace tensor_utils
}  // namespace dicp
