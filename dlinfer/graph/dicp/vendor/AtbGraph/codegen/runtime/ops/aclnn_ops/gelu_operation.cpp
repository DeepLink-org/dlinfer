#include "gelu_operation.h"

#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "acl/acl.h"
#include "aclnnop/aclnn_gelu.h"
#include "utils/common.h"
#include "utils/log.h"
#include "utils/tensor_utils.h"

namespace dicp {
const int NUM1 = 1;

AclNnGeluOperation::AclNnGeluOperation(const std::string& name) : AclNnOperation(name) {}

AclNnGeluOperation::~AclNnGeluOperation() {}

atb::Status AclNnGeluOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnGeluOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnGeluOperation::GetOutputNum() const { return NUM1; }

int AclNnGeluOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnGeluGetWorkspaceSize start";

    int ret = aclnnGeluGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnGeluGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnGeluOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnGelu start";
    int ret = aclnnGelu(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnGelu end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnGeluOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnGeluOperation: name: " << opName;
    atb::Operation* op = new AclNnGeluOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnGeluOperation, AclNnGeluOperationCreate);

}  // namespace dicp
