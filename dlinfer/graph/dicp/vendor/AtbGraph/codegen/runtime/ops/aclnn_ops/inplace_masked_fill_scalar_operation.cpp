#include "inplace_masked_fill_scalar_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnInplaceMaskedFillScalar::AclNnInplaceMaskedFillScalar(const std::string& name, float value, const std::string& dtype) : AclNnOperation(name) {
    value_ = DICPScalar(value, dtype);
    aclValue_ = aclCreateScalar(value_.getValuePtr(), value_.getDataType());
}

AclNnInplaceMaskedFillScalar::~AclNnInplaceMaskedFillScalar() {
    if (aclValue_ != nullptr) {
        aclDestroyScalar(aclValue_);
    }
}

atb::Status AclNnInplaceMaskedFillScalar::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceMaskedFillScalar::GetInputNum() const { return NUM2; }

uint32_t AclNnInplaceMaskedFillScalar::GetOutputNum() const { return NUM1; }

int AclNnInplaceMaskedFillScalar::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclValue_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceDivGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceMaskedFillScalar::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnInplaceMaskedFillScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclnnInplaceMaskedFillScalar end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceMaskedFillScalarCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnInplaceMaskedFillScalar: name: " << opName << ", value: " << value << ", dtype: " << dtype;
    atb::Operation* op = new AclNnInplaceMaskedFillScalar(opName, value, dtype);
    return op;
}

REGISTER_OPERATION(AclNnInplaceMaskedFillScalar, AclNnInplaceMaskedFillScalarCreate);

}  // namespace dicp
