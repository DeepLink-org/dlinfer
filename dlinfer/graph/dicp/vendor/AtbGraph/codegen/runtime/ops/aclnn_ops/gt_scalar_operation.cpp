#include "gt_scalar_operation.h"

#include "aclnnop/aclnn_gt_scalar.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnGtScalarOperation::AclNnGtScalarOperation(const std::string& name, float value, const std::string& dtype) : AclNnOperation(name) {
    other_ = DICPScalar(value, dtype);
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
}

AclNnGtScalarOperation::~AclNnGtScalarOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
}

atb::Status AclNnGtScalarOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = aclDataType::ACL_BOOL;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnGtScalarOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnGtScalarOperation::GetOutputNum() const { return NUM1; }

int AclNnGtScalarOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnGtScalarGetWorkspaceSize start";

    int ret = aclnnGtScalarGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnGtScalarGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnGtScalarOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnGtScalar start";
    int ret = aclnnGtScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnGtScalar end, ret:" << ret;
    return ret;
}

}  // namespace dicp
