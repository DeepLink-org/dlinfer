#include "utils/tensor_utils.h"

#include <acl/acl.h>
#include <atb/utils.h>
#include <sys/stat.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "utils/log.h"

namespace dicp {
namespace tensor_utils {

static int64_t GetTensorNpuFormat(const at::Tensor& tensor) { return at_npu::native::get_npu_format(tensor); }

std::string TensorToString(const atb::Tensor& tensor) {
    std::stringstream ss;
    ss << TensorDescToString(tensor.desc) << ", deviceData:" << tensor.deviceData << ", hostData:" << tensor.hostData << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorDescToString(const atb::TensorDesc& tensorDesc) {
    std::stringstream ss;
    ss << "dtype: " << tensorDesc.dtype << ", format: " << tensorDesc.format << ", shape:[";
    for (size_t i = 0; i < tensorDesc.shape.dimNum; ++i) {
        if (i == 0) {
            ss << tensorDesc.shape.dims[i];
        } else {
            ss << ", " << tensorDesc.shape.dims[i];
        }
    }
    ss << "]";

    return ss.str();
}

atb::Tensor AtTensor2Tensor(const at::Tensor& atTensor) {
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},
        {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT},
        {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},
        {at::ScalarType::BFloat16, ACL_BF16},
    };

    DICP_LOG_IF(!atTensor.is_contiguous(), ERROR) << "atTensor is not contiguous";
    atb::Tensor tensor;
    tensor.desc.format = static_cast<aclFormat>(GetTensorNpuFormat(atTensor));
    tensor.deviceData = atTensor.data_ptr();

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    if (tensor.desc.shape.dimNum == 0) {
        tensor.desc.shape.dimNum = 1;
        tensor.desc.shape.dims[0] = 1;
    }

    if (tensor.desc.shape.dimNum == 1 && tensor.desc.shape.dims[0] == 0) {
        tensor.desc.shape.dimNum = 0;
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        tensor.desc.dtype = it->second;
    } else {
        DICP_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}

at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc& tensorDesc) {
    static std::map<aclDataType, at::ScalarType> dtypeMap = {
        {ACL_BOOL, at::ScalarType::Bool},
        {ACL_UINT8, at::ScalarType::Byte},
        {ACL_INT8, at::ScalarType::Char},
        {ACL_FLOAT16, at::ScalarType::Half},
        {ACL_FLOAT, at::ScalarType::Float},
        {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long},
        {ACL_BF16, at::ScalarType::BFloat16},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = dtypeMap.find(tensorDesc.dtype);
    if (it != dtypeMap.end()) {
        options = options.dtype(it->second);
    } else {
        DICP_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

    options = options.layout(torch::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());

    DICP_LOG(INFO) << "tensor_with_format stat, " << TensorDescToString(tensorDesc);

    at::Tensor newTensor = at_npu::native::empty_with_format(at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);

    DICP_LOG(INFO) << "tensor_with_format end, newTensor.format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous();
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        DICP_LOG(WARN) << "tensor_with_format newTensor.format:" << GetTensorNpuFormat(newTensor) << " != " << tensorDesc.format;
        newTensor = at_npu::native::npu_format_cast(newTensor, tensorDesc.format);
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    DICP_LOG(INFO) << "tensor_with_format success, newTensor.options:" << newTensor.options() << ", format:" << GetTensorNpuFormat(newTensor)
                   << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

int64_t TransferAtTensor2AtbTensor(std::vector<torch::Tensor>& atTensors, std::vector<atb::Tensor>& atbTensors) {
    for (auto& atTensor : atTensors) {
        if (!atTensor.is_contiguous()) {
            atTensor = atTensor.contiguous();
        }
        atb::Tensor tensor = AtTensor2Tensor(atTensor);
        atbTensors.push_back(tensor);
    }
    return atb::NO_ERROR;
}

}  // namespace tensor_utils
}  // namespace dicp
