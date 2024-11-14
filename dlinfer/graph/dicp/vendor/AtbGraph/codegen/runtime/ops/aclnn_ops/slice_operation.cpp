#include "slice_operation.h"

#include "aclnnop/aclnn_slice.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnSliceOperation::AclNnSliceOperation(const std::string& name, int64_t dim, int64_t start, int64_t end, int64_t step)
    : AclNnOperation(name), dim_(dim), start_(start), end_(end), step_(step) {}

AclNnSliceOperation::~AclNnSliceOperation() {}

atb::Status AclNnSliceOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    int64_t sliceNum = (end_ - start_) / step_;
    outTensorDescs.at(0).shape.dims[dim_] = sliceNum;
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnSliceOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnSliceOperation::GetOutputNum() const { return NUM1; }

int AclNnSliceOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnSliceGetWorkspaceSize start";

    int ret = aclnnSliceGetWorkspaceSize(aclInTensors_.at(0).tensor, dim_, start_, end_, step_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnSliceGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSliceOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnSlice start";
    int ret = aclnnSlice(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnSlice end, ret:" << ret;
    return ret;
}

}  // namespace dicp
