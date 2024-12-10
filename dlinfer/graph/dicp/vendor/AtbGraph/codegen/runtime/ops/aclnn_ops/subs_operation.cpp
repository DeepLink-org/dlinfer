#include "subs_operation.h"

#include "aclnnop/aclnn_sub.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnSubsOperation::AclNnSubsOperation(const std::string& name, float value, float alpha, const std::string& dtype) : AclNnOperation(name) {
    other_ = DICPScalar(value, dtype);
    alpha_ = DICPScalar(alpha, dtype);
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

AclNnSubsOperation::~AclNnSubsOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
}

atb::Status AclNnSubsOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnSubsOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnSubsOperation::GetOutputNum() const { return NUM1; }

int AclNnSubsOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnSubsGetWorkspaceSize start";

    int ret = aclnnSubsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclAlpha_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnSubsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSubsOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnSubs start";
    int ret = aclnnSubs(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnSubs end, ret:" << ret;
    return ret;
}

}  // namespace dicp
