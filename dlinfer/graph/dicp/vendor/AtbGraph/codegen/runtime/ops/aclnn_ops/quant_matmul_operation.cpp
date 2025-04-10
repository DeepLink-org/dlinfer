#include "quant_matmul_operation.h"

#include "aclnnop/aclnn_quant_matmul_v4.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM4 = 4;
const int NUM5 = 5;

AclNnQuantMatmulOperation::AclNnQuantMatmulOperation(const std::string& name, bool hasBias) : AclNnOperation(name), hasBias_(hasBias) {}

AclNnQuantMatmulOperation::~AclNnQuantMatmulOperation() {}

atb::Status AclNnQuantMatmulOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = aclDataType::ACL_FLOAT16;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0];

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnQuantMatmulOperation::GetInputNum() const { return hasBias_ ? NUM5 : NUM4; }

uint32_t AclNnQuantMatmulOperation::GetOutputNum() const { return NUM1; }

int AclNnQuantMatmulOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnQuantMatmulV4GetWorkspaceSize start";
    auto bias = hasBias_ ? aclInTensors_.at(4).tensor : nullptr;
    int ret = aclnnQuantMatmulV4GetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                 aclInTensors_.at(1).tensor,
                                                 aclInTensors_.at(2).tensor,
                                                 nullptr,
                                                 aclInTensors_.at(3).tensor,
                                                 bias,
                                                 false,
                                                 true,
                                                 aclOutTensors_.at(0).tensor,
                                                 &workspaceSize,
                                                 &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnQuantMatmulV4GetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnQuantMatmulOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnQuantMatmulV4 start";
    int ret = aclnnQuantMatmulV4(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnQuantMatmulV4 end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnQuantMatmulOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    bool hasBias;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("hasBias")) {
        hasBias = paramJson["hasBias"].get<bool>();
    }
    DICP_LOG(INFO) << "AclNnQuantMatmulOperation: name: " << opName << ", hasBias:" << hasBias;
    atb::Operation* op = new AclNnQuantMatmulOperation(opName, hasBias);
    return op;
}

REGISTER_OPERATION(AclNnQuantMatmulOperation, AclNnQuantMatmulOperationCreate);

}  // namespace dicp
