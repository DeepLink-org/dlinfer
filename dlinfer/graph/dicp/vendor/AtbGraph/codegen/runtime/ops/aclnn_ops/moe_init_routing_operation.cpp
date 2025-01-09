#include "moe_init_routing_operation.h"

#include "aclnnop/aclnn_moe_init_routing.h"
// #include "aclnnop/aclnn_moe_init_routing_v2.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;

AclNnMoeInitRoutingOperation::AclNnMoeInitRoutingOperation(const std::string& name, int64_t activeNum, int64_t numExperts)
    : AclNnOperation(name), activeNum_(activeNum), numExperts_(numExperts) {}

AclNnMoeInitRoutingOperation::~AclNnMoeInitRoutingOperation() {}

atb::Status AclNnMoeInitRoutingOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    auto seqLength = inTensorDescs.at(2).shape.dims[0];
    auto topk = inTensorDescs.at(2).shape.dims[1];

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] * topk;
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

    outTensorDescs.at(1).format = inTensorDescs.at(1).format;
    outTensorDescs.at(1).shape.dimNum = NUM1;
    outTensorDescs.at(1).dtype = inTensorDescs.at(1).dtype;
    outTensorDescs.at(1).shape.dims[0] = seqLength * topk;

    outTensorDescs.at(2).format = inTensorDescs.at(2).format;
    outTensorDescs.at(2).shape.dimNum = NUM1;
    outTensorDescs.at(2).dtype = inTensorDescs.at(2).dtype;
    outTensorDescs.at(2).shape.dims[0] = seqLength * topk;

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnMoeInitRoutingOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnMoeInitRoutingOperation::GetOutputNum() const { return NUM3; }

int AclNnMoeInitRoutingOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeInitRoutingGetWorkspaceSize start";

    int ret = aclnnMoeInitRoutingGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                  aclInTensors_.at(1).tensor,
                                                  aclInTensors_.at(2).tensor,
                                                  activeNum_,
                                                  aclOutTensors_.at(0).tensor,
                                                  aclOutTensors_.at(1).tensor,
                                                  aclOutTensors_.at(2).tensor,
                                                  &workspaceSize,
                                                  &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMoeInitRoutingGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMoeInitRoutingOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnMoeInitRouting start";
    int ret = aclnnMoeInitRouting(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnMoeInitRouting end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnMoeInitRoutingOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t activeNum;
    int64_t numExperts;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("activeNum")) {
        activeNum = paramJson["activeNum"].get<int64_t>();
    }
    if (paramJson.contains("numExperts")) {
        numExperts = paramJson["numExperts"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnMoeInitRoutingOperation: name: " << opName << " activeNum:" << activeNum << " numExperts:" << numExperts;
    atb::Operation* op = new AclNnMoeInitRoutingOperation(opName, activeNum, numExperts);
    return op;
}

REGISTER_OPERATION(AclNnMoeInitRoutingOperation, AclNnMoeInitRoutingOperationCreate);

}  // namespace dicp
