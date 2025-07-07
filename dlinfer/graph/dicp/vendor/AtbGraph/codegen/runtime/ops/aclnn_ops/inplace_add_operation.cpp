#include "inplace_add_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_add.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnInplaceAddOperation::AclNnInplaceAddOperation(const std::string& name, float aplpha, const std::string& dtype) : AclNnOperation(name) {
    alpha_ = DICPScalar(aplpha, dtype);
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

AclNnInplaceAddOperation::~AclNnInplaceAddOperation() {
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
}

atb::Status AclNnInplaceAddOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceAddOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnInplaceAddOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceAddOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnInplaceAddGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclAlpha_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceAddGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceAddOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnInplaceAdd(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceAdd end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceAddOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float aplpha;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("aplpha")) {
        aplpha = paramJson["aplpha"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnInplaceAddOperation: name: " << opName;
    atb::Operation* op = new AclNnInplaceAddOperation(opName, aplpha, dtype);
    return op;
}

REGISTER_OPERATION(AclNnInplaceAddOperation, AclNnInplaceAddOperationCreate);

}  // namespace dicp
