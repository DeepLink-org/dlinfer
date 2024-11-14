#include "s_where_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_s_where.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM3 = 3;

AclNnSWhereOperation::AclNnSWhereOperation(const std::string& name) : AclNnOperation(name) {}

AclNnSWhereOperation::~AclNnSWhereOperation() {}

atb::Status AclNnSWhereOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(1).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
    auto dimNum0 = inTensorDescs.at(0).shape.dimNum;
    auto dimNum1 = inTensorDescs.at(1).shape.dimNum;
    auto dimNum2 = inTensorDescs.at(2).shape.dimNum;
    outTensorDescs.at(0).shape.dimNum = std::max({dimNum0, dimNum1, dimNum2});
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        auto dim0 = i < inTensorDescs.at(0).shape.dimNum ? inTensorDescs.at(0).shape.dims[i] : -1;
        auto dim1 = i < inTensorDescs.at(1).shape.dimNum ? inTensorDescs.at(1).shape.dims[i] : -1;
        auto dim2 = i < inTensorDescs.at(2).shape.dimNum ? inTensorDescs.at(2).shape.dims[i] : -1;
        outTensorDescs.at(0).shape.dims[i] = std::max({dim0, dim1, dim2});
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnSWhereOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnSWhereOperation::GetOutputNum() const { return NUM1; }

int AclNnSWhereOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnSWhereGetWorkspaceSize start";
    int ret = aclnnSWhereGetWorkspaceSize(
        aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclInTensors_.at(2).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnSWhereGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnSWhereOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnSWhere start";
    int ret = aclnnSWhere(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnSWhere end, ret:" << ret;
    return ret;
}

}  // namespace dicp
