#include "inplace_index_copy_operation.h"

#include "aclnnop/aclnn_index_copy.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM3 = 3;

AclNnInplaceIndexCopyOperation::AclNnInplaceIndexCopyOperation(const std::string& name, int64_t dim) : AclNnOperation(name), dim_(dim) {}

AclNnInplaceIndexCopyOperation::~AclNnInplaceIndexCopyOperation() {}

atb::Status AclNnInplaceIndexCopyOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                       atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceIndexCopyOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnInplaceIndexCopyOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceIndexCopyOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceIndexCopyGetWorkspaceSize start";
    int ret = aclnnInplaceIndexCopyGetWorkspaceSize(
        aclInTensors_.at(0).tensor, dim_, aclInTensors_.at(1).tensor, aclInTensors_.at(2).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceIndexCopyGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceIndexCopyOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceIndexCopy start";
    int ret = aclnnInplaceIndexCopy(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceIndexCopy end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceIndexCopyOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<std::int64_t>();
    }
    DICP_LOG(INFO) << "AclNnInplaceIndexCopyOperation: name: " << opName << ", dim: " << dim;
    atb::Operation* op = new AclNnInplaceIndexCopyOperation(opName, dim);
    return op;
}

REGISTER_OPERATION(AclNnInplaceIndexCopyOperation, AclNnInplaceIndexCopyOperationCreate);

}  // namespace dicp
