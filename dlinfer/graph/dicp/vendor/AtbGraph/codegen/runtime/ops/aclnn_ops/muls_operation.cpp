#include "muls_operation.h"

#include "aclnnop/aclnn_mul.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnMulsOperation::AclNnMulsOperation(const std::string& name, float value, const std::string& dtype) : AclNnOperation(name) {
    other_ = DICPScalar(value, dtype);
    aclOther_ = aclCreateScalar(other_.getValuePtr(), other_.getDataType());
}

AclNnMulsOperation::~AclNnMulsOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
}

atb::Status AclNnMulsOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnMulsOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnMulsOperation::GetOutputNum() const { return NUM1; }

int AclNnMulsOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize start";

    int ret = aclnnMulsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMulsOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMuls start";
    int ret = aclnnMuls(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMuls end, ret:" << ret;
    return ret;
}

}  // namespace dicp
