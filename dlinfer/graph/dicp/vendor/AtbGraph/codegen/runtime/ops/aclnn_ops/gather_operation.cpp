#include "gather_operation.h"

#include <algorithm>
#include <cstdint>

#include "aclnnop/aclnn_gather.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnGatherOperation::AclNnGatherOperation(const std::string& name, int64_t dim) : AclNnOperation(name), dim_(dim) {}

AclNnGatherOperation::~AclNnGatherOperation() {}

atb::Status AclNnGatherOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(1).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnGatherOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnGatherOperation::GetOutputNum() const { return NUM1; }

int AclNnGatherOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnGatherGetWorkspaceSize start";

    int ret =
        aclnnGatherGetWorkspaceSize(aclInTensors_.at(0).tensor, dim_, aclInTensors_.at(1).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnGatherGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnGatherOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnGather start";
    int ret = aclnnGather(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnGather end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnGatherOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnGatherOperation: name: " << opName;
    atb::Operation* op = new AclNnGatherOperation(opName, dim);
    return op;
}

REGISTER_OPERATION(AclNnGatherOperation, AclNnGatherOperationCreate);

}  // namespace dicp
