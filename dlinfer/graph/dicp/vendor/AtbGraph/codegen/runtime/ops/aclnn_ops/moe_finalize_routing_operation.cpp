#include "moe_finalize_routing_operation.h"

#include "aclnnop/aclnn_moe_finalize_routing.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM7 = 7;

AclNnMoeFinalizeRoutingOperation::AclNnMoeFinalizeRoutingOperation(const std::string& name) : AclNnOperation(name) {}

AclNnMoeFinalizeRoutingOperation::~AclNnMoeFinalizeRoutingOperation() {}

atb::Status AclNnMoeFinalizeRoutingOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                         atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(1).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnMoeFinalizeRoutingOperation::GetInputNum() const { return NUM7; }

uint32_t AclNnMoeFinalizeRoutingOperation::GetOutputNum() const { return NUM1; }

int AclNnMoeFinalizeRoutingOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeFinalizeRoutingGetWorkspaceSize start";

    int ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                      aclInTensors_.at(1).tensor,
                                                      aclInTensors_.at(2).tensor,
                                                      aclInTensors_.at(3).tensor,
                                                      aclInTensors_.at(4).tensor,
                                                      aclInTensors_.at(5).tensor,
                                                      aclInTensors_.at(6).tensor,
                                                      aclOutTensors_.at(0).tensor,
                                                      &workspaceSize,
                                                      &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMoeFinalizeRoutingGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMoeFinalizeRoutingOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeFinalizeRouting start";
    int ret = aclnnMoeFinalizeRouting(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMoeFinalizeRouting end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnMoeFinalizeRoutingOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    atb::Operation* op = new AclNnMoeFinalizeRoutingOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnMoeFinalizeRoutingOperation, AclNnMoeFinalizeRoutingOperationCreate);

}  // namespace dicp
