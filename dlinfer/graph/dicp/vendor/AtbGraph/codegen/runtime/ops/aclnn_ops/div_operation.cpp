#include "div_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_div.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnDivOperation::AclNnDivOperation(const std::string& name) : AclNnOperation(name) {}

AclNnDivOperation::~AclNnDivOperation() {}

atb::Status AclNnDivOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
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

uint32_t AclNnDivOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnDivOperation::GetOutputNum() const { return NUM1; }

int AclNnDivOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnDivGetWorkspaceSize start";

    int ret = aclnnDivGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnDivGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnDivOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnDiv start";
    int ret = aclnnDiv(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnDiv end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnDivOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnDivOperation: name: " << opName;
    atb::Operation* op = new AclNnDivOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnDivOperation, AclNnDivOperationCreate);

}  // namespace dicp
