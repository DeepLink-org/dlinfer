#include "divs_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_div.h"
#include "log.h"

namespace dicp {

const int NUM1 = 1;

AclNnDivsOperation::AclNnDivsOperation(const std::string& name, float divisor) : AclNnOperation(name) {
    divisor_ = aclCreateScalar(&divisor, aclDataType::ACL_FLOAT);
}

AclNnDivsOperation::~AclNnDivsOperation() {}

atb::Status AclNnDivsOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnDivsOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnDivsOperation::GetOutputNum() const { return NUM1; }

int AclNnDivsOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnDivsGetWorkspaceSize start";

    int ret = aclnnDivsGetWorkspaceSize(aclInTensors_.at(0).tensor, divisor_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnDivsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnDivsOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnDivs start";
    int ret = aclnnDivs(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnDivs end, ret:" << ret;
    return ret;
}

}  // namespace dicp
