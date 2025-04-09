#include "masked_fill_scalar_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "aclnnop/aclnn_mul.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

MaskedFillScalarOperation::MaskedFillScalarOperation(const std::string& name, float value, const std::string& dtype) : AclNnOperation(name) {
    value_ = DICPScalar(value, dtype);
    aclValue_ = aclCreateScalar(value_.getValuePtr(), value_.getDataType());
    one_ = DICPScalar(1.0, dtype);
    aclOne_ = aclCreateScalar(one_.getValuePtr(), one_.getDataType());
}

MaskedFillScalarOperation::~MaskedFillScalarOperation() {
    if (aclValue_ != nullptr) {
        aclDestroyScalar(aclValue_);
    }
    if (aclOne_ != nullptr) {
        aclDestroyScalar(aclOne_);
    }
}

atb::Status MaskedFillScalarOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t MaskedFillScalarOperation::GetInputNum() const { return NUM2; }

uint32_t MaskedFillScalarOperation::GetOutputNum() const { return NUM1; }

int MaskedFillScalarOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnMulsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOne_, aclOutTensors_.at(0).tensor, &mulsWorkspaceSize_, &aclMulsExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << mulsWorkspaceSize_ << ", aclExecutor:" << aclMulsExecutor_;

    ret = aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclOutTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclValue_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceDivGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int MaskedFillScalarOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnMuls(workspace, mulsWorkspaceSize_, aclMulsExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMuls end, ret:" << ret;

    ret = aclnnInplaceMaskedFillScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMaskedFillScalar end, ret:" << ret;
    return ret;
}

atb::Operation* MaskedFillScalarOperationCreate(const nlohmann::json& paramJson) {
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
    DICP_LOG(INFO) << "MaskedFillScalarOperation: name: " << opName << ", value: " << value << ", dtype: " << dtype;
    atb::Operation* op = new MaskedFillScalarOperation(opName, value, dtype);
    return op;
}

REGISTER_OPERATION(MaskedFillScalarOperation, MaskedFillScalarOperationCreate);

}  // namespace dicp
