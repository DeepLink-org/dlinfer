#include "softmax_operation.h"

#include "aclnnop/aclnn_softmax.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnSoftmaxOperation::AclNnSoftmaxOperation(const std::string& name, int64_t dim) : AclNnOperation(name), dim_(dim) {}

AclNnSoftmaxOperation::~AclNnSoftmaxOperation() {}

atb::Status AclNnSoftmaxOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnSoftmaxOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnSoftmaxOperation::GetOutputNum() const { return NUM1; }

int AclNnSoftmaxOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnSoftmaxGetWorkspaceSize start";

    int ret = aclnnSoftmaxGetWorkspaceSize(aclInTensors_.at(0).tensor, dim_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnSoftmaxGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSoftmaxOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnSoftmax start";
    int ret = aclnnSoftmax(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnSoftmax end, ret:" << ret;
    return ret;
}

}  // namespace dicp
