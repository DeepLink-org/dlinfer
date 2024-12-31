#include "bincount_operation.h"

#include "aclnnop/aclnn_bincount.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnBincountOperation::AclNnBincountOperation(const std::string& name, int64_t minlength) : AclNnOperation(name), minlength_(minlength) {}

AclNnBincountOperation::~AclNnBincountOperation() {}

atb::Status AclNnBincountOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(0).shape.dims[0] = minlength_;
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnBincountOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnBincountOperation::GetOutputNum() const { return NUM1; }

int AclNnBincountOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnBincountOperationGetWorkspaceSize start";

    int ret = aclnnBincountGetWorkspaceSize(aclOutTensors_.at(0).tensor, nullptr, minlength_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnBincountGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnBincountOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnBincount start";
    int ret = aclnnBincount(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnBincount end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnBincountOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t minlength;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("minlength")) {
        minlength = paramJson["minlength"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnBincountOperation: name: " << opName << " minlength:" << minlength;
    atb::Operation* op = new AclNnBincountOperation(opName, minlength);
    return op;
}

REGISTER_OPERATION(AclNnBincountOperation, AclNnBincountOperationCreate);

}  // namespace dicp
