#include "adds_operation.h"

#include <iostream>

#include "aclnnop/aclnn_add.h"
#include "utils/log.h"
#include "utils/misc.h"
#include "utils/scalar.h"

namespace dicp {

const int NUM1 = 1;

AclNnAddsOperation::AclNnAddsOperation(const std::string& name, float value, float alpha, const std::string& dtype) : AclNnOperation(name) {
    other_ = DICPScalar(value, dtype);
    alpha_ = DICPScalar(alpha, dtype);
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

AclNnAddsOperation::~AclNnAddsOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
}

atb::Status AclNnAddsOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnAddsOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnAddsOperation::GetOutputNum() const { return NUM1; }

int AclNnAddsOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnAddsGetWorkspaceSize start";
    int ret = aclnnAddsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclAlpha_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnAddsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnAddsOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnAdds start";
    int ret = aclnnAdds(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnAdds end, ret:" << ret;
    return ret;
}

}  // namespace dicp
