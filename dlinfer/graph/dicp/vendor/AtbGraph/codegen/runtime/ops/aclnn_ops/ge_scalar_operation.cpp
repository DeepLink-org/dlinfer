#include "ge_scalar_operation.h"

#include "aclnnop/aclnn_ge_scalar.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnGeScalarOperation::AclNnGeScalarOperation(const std::string& name, float value, const std::string& dtype) : AclNnOperation(name) {
    other_ = DICPScalar(value, dtype);
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
}

AclNnGeScalarOperation::~AclNnGeScalarOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
}

atb::Status AclNnGeScalarOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = aclDataType::ACL_BOOL;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnGeScalarOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnGeScalarOperation::GetOutputNum() const { return NUM1; }

int AclNnGeScalarOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnGeScalarGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnGeScalarGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnGeScalarOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnGeScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnGeScalar end, ret:" << ret;
    return ret;
}

}  // namespace dicp
