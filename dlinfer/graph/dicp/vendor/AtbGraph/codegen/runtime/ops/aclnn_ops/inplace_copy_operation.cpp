#include "inplace_copy_operation.h"

#include "aclnnop/aclnn_copy.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnInplaceCopyOperation::AclNnInplaceCopyOperation(const std::string& name) : AclNnOperation(name) {}

AclNnInplaceCopyOperation::~AclNnInplaceCopyOperation() {}

atb::Status AclNnInplaceCopyOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceCopyOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnInplaceCopyOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceCopyOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceCopyGetWorkspaceSize start";
    int ret = aclnnInplaceCopyGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceCopyGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceCopyOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceCopy start";
    int ret = aclnnInplaceCopy(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceCopy end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceCopyOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnInplaceCopyOperation: name: " << opName;
    atb::Operation* op = new AclNnInplaceCopyOperation(opName);
    return op;
}

REGISTER_OPERATION(AclNnInplaceCopyOperation, AclNnInplaceCopyOperationCreate);

}  // namespace dicp
