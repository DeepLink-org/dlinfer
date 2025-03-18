#include "dynamic_quant_operation.h"

#include "aclnnop/aclnn_dynamic_quant.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnDynamicQuantOperation::AclNnDynamicQuantOperation(const std::string& name) : AclNnOperation(name) {}

AclNnDynamicQuantOperation::~AclNnDynamicQuantOperation() {}

atb::Status AclNnDynamicQuantOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = aclDataType::ACL_INT8;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    outTensorDescs.at(1).format = inTensorDescs.at(0).format;
    outTensorDescs.at(1).shape.dimNum = inTensorDescs.at(0).shape.dimNum - 1;
    outTensorDescs.at(1).dtype = aclDataType::ACL_FLOAT;
    for (size_t i = 0; i < outTensorDescs.at(1).shape.dimNum; ++i) {
        outTensorDescs.at(1).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnDynamicQuantOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnDynamicQuantOperation::GetOutputNum() const { return NUM2; }

int AclNnDynamicQuantOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnDynamicQuantGetWorkspaceSize start";
    int ret = aclnnDynamicQuantGetWorkspaceSize(
        aclInTensors_.at(0).tensor, nullptr, aclOutTensors_.at(0).tensor, aclOutTensors_.at(1).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnDynamicQuantGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnDynamicQuantOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnDynamicQuant start";
    int ret = aclnnDynamicQuant(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnDynamicQuant end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnDynamicQuantOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnDynamicQuantOperation: name: " << opName;
    atb::Operation* op = new AclNnDynamicQuantOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnDynamicQuantOperation, AclNnDynamicQuantOperationCreate);

}  // namespace dicp
