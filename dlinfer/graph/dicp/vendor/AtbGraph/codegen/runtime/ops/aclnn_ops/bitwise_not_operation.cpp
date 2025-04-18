#include "bitwise_not_operation.h"

#include "aclnnop/level2/aclnn_bitwise_not.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnBitwiseNotOperation::AclNnBitwiseNotOperation(const std::string& name) : AclNnOperation(name) {}

AclNnBitwiseNotOperation::~AclNnBitwiseNotOperation() {}

atb::Status AclNnBitwiseNotOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnBitwiseNotOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnBitwiseNotOperation::GetOutputNum() const { return NUM1; }

int AclNnBitwiseNotOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnBitwiseNotGetWorkspaceSize start";
    int ret = aclnnBitwiseNotGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnBitwiseNotGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnBitwiseNotOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnBitwiseNot start";
    int ret = aclnnBitwiseNot(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnBitwiseNot end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnBitwiseNotOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnBitwiseNotOperation name: " << opName;
    atb::Operation* op = new AclNnBitwiseNotOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnBitwiseNotOperation, AclNnBitwiseNotOperationCreate);

}  // namespace dicp
