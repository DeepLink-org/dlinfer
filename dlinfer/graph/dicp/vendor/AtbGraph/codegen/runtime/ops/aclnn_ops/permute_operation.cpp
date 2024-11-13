#include "permute_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_permute.h"
#include "utils/log.h"

namespace dicp {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

AclNnPermuteOperation::AclNnPermuteOperation(const std::string& name, std::vector<int64_t> dims) : AclNnOperation(name), dims_(std::move(dims)) {}

AclNnPermuteOperation::~AclNnPermuteOperation() {}

atb::Status AclNnPermuteOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    for (size_t i = 0; i < dims_.size(); ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[dims_[i]];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnPermuteOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnPermuteOperation::GetOutputNum() const { return NUM1; }

int AclNnPermuteOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize start";
    aclIntArray* dims = aclCreateIntArray(dims_.data(), dims_.size());
    int ret = aclnnPermuteGetWorkspaceSize(aclInTensors_.at(0).tensor, dims, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnPermuteOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnPermute start";
    int ret = aclnnPermute(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnPermute end, ret:" << ret;
    return ret;
}

}  // namespace dicp
