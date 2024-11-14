#include "topk_operation.h"

#include "aclnnop/aclnn_topk.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnTopkOperation::AclNnTopkOperation(const std::string& name, int64_t k, int64_t dim) : AclNnOperation(name), k_(k), dim_(dim) {}

AclNnTopkOperation::~AclNnTopkOperation() {}

atb::Status AclNnTopkOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    int64_t dim = dim_ < 0 ? dim_ + inTensorDescs.at(0).shape.dimNum : dim_;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(1).format = inTensorDescs.at(0).format;
    outTensorDescs.at(1).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(1).dtype = aclDataType::ACL_INT64;

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        if (i == dim) {
            outTensorDescs.at(0).shape.dims[dim] = k_;
            outTensorDescs.at(1).shape.dims[dim] = k_;
        } else {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
            outTensorDescs.at(1).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        }
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnTopkOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnTopkOperation::GetOutputNum() const { return NUM2; }

int AclNnTopkOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnTopkGetWorkspaceSize start";

    int ret = aclnnTopkGetWorkspaceSize(
        aclInTensors_.at(0).tensor, k_, true, true, dim_, aclOutTensors_.at(0).tensor, aclOutTensors_.at(1).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnTopkGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnTopkOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnTopk start";
    int ret = aclnnTopk(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnTopk end, ret:" << ret;
    return ret;
}

}  // namespace dicp
