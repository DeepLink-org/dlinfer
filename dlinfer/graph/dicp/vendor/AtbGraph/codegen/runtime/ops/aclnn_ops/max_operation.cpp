#include "max_operation.h"

#include "aclnnop/aclnn_max.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnMaxOperation::AclNnMaxOperation(const std::string& name) : AclNnOperation(name) {}

AclNnMaxOperation::~AclNnMaxOperation() {}

atb::Status AclNnMaxOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = NUM1;
    outTensorDescs.at(0).shape.dims[0] = 1;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnMaxOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnMaxOperation::GetOutputNum() const { return NUM1; }

int AclNnMaxOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMaxGetWorkspaceSize start";
    int ret = aclnnMaxGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMaxGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMaxOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMax start";
    int ret = aclnnMax(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMax end, ret:" << ret;
    return ret;
}

}  // namespace dicp
