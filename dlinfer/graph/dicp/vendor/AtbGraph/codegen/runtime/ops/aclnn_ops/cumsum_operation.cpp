#include "cumsum_operation.h"

#include "aclnnop/aclnn_cumsum.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnCumsumOperation::AclNnCumsumOperation(const std::string& name, int64_t dim, aclDataType dtype) : AclNnOperation(name), dim_(dim), dtype_(dtype) {}

AclNnCumsumOperation::~AclNnCumsumOperation() {}

atb::Status AclNnCumsumOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = outTensorDescs.at(0).dtype;

    auto rank = inTensorDescs.at(0).shape.dimNum;
    for (size_t i = 0; i < rank; ++i) {
        outTensorDescs.at(0).shape.dims[i] = i == (dim_ + rank) % rank ? NUM1 : inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnCumsumOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnCumsumOperation::GetOutputNum() const { return NUM1; }

int AclNnCumsumOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnCumsumGetWorkspaceSize start";

    int ret = aclnnCumsumGetWorkspaceSize(aclOutTensors_.at(0).tensor, dim_, dtype_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnCumsumGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnCumsumOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnCumsum start";
    int ret = aclnnCumsum(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnCumsum end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnCumsumOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    if (paramJson.contains("outTensorType")) {
        dataType = static_cast<aclDataType>(paramJson["outTensorType"].get<int32_t>());
    }
    DICP_LOG(INFO) << "AclNnBincountOperation: name: " << opName << " dim:" << dim << " dtype:" << dataType;
    atb::Operation* op = new AclNnCumsumOperation(opName, dim, dataType);
    return op;
}

REGISTER_OPERATION(AclNnCumsumOperation, AclNnCumsumOperationCreate);

}  // namespace dicp
