#include "index_select_operation.h"

#include "aclnnop/aclnn_index_select.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnIndexSelectOperation::AclNnIndexSelectOperation(const std::string& name, int64_t dim) : AclNnOperation(name), dim_(dim) {}

AclNnIndexSelectOperation::~AclNnIndexSelectOperation() {}

atb::Status AclNnIndexSelectOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    outTensorDescs.at(0).shape.dims[dim_] = inTensorDescs.at(1).shape.dims[0];
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnIndexSelectOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnIndexSelectOperation::GetOutputNum() const { return NUM1; }

int AclNnIndexSelectOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnIndexSelectGetWorkspaceSize start";

    int ret = aclnnIndexSelectGetWorkspaceSize(
        aclInTensors_.at(0).tensor, dim_, aclInTensors_.at(1).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnIndexSelectGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnIndexSelectOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnIndexSelect start";
    int ret = aclnnIndexSelect(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnIndexSelect end, ret:" << ret;
    return ret;
}

}  // namespace dicp
