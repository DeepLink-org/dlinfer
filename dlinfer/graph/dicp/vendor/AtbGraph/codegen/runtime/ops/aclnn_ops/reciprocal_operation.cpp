#include "reciprocal_operation.h"

#include "aclnnop/aclnn_reciprocal.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnReciprocalOperation::AclNnReciprocalOperation(const std::string& name) : AclNnOperation(name) {}

AclNnReciprocalOperation::~AclNnReciprocalOperation() {}

atb::Status AclNnReciprocalOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnReciprocalOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnReciprocalOperation::GetOutputNum() const { return NUM1; }

int AclNnReciprocalOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnReciprocalGetWorkspaceSize start";
    int ret = aclnnReciprocalGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnReciprocalGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnReciprocalOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnReciprocal start";
    int ret = aclnnReciprocal(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnReciprocal end, ret:" << ret;
    return ret;
}

}  // namespace dicp
