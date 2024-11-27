#include "sub_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_sub.h"
#include "utils/log.h"
#include "utils/misc.h"
#include "utils/scalar.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnSubOperation::AclNnSubOperation(const std::string& name, float alpha, const std::string& dtype) : AclNnOperation(name) {
    alpha_ = DICPScalar(alpha, dtype);
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

AclNnSubOperation::~AclNnSubOperation() {
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
}

atb::Status AclNnSubOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = std::max(inTensorDescs.at(0).shape.dimNum, inTensorDescs.at(1).shape.dimNum);

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        auto dim0 = i < inTensorDescs.at(0).shape.dimNum ? inTensorDescs.at(0).shape.dims[i] : -1;
        auto dim1 = i < inTensorDescs.at(1).shape.dimNum ? inTensorDescs.at(1).shape.dims[i] : -1;
        outTensorDescs.at(0).shape.dims[i] = std::max({dim0, dim1});
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnSubOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnSubOperation::GetOutputNum() const { return NUM1; }

int AclNnSubOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnSubGetWorkspaceSize start";
    int ret =
        aclnnSubGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclAlpha_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnSubGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSubOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnSub start";
    int ret = aclnnSub(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnSub end, ret:" << ret;
    return ret;
}

}  // namespace dicp
