#include "inplace_div_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_div.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnInplaceDivOperation::AclNnInplaceDivOperation(const std::string& name) : AclNnOperation(name) {}

AclNnInplaceDivOperation::~AclNnInplaceDivOperation() {}

atb::Status AclNnInplaceDivOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceDivOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnInplaceDivOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceDivOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnInplaceDivGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceDivGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceDivOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnInplaceDiv(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnInplaceDiv end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceDivOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float divisor;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnInplaceDivOperation: name: " << opName;
    atb::Operation* op = new AclNnInplaceDivOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnInplaceDivOperation, AclNnInplaceDivOperationCreate);

}  // namespace dicp
