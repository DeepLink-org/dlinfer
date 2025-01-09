#include "moe_token_unpermute_operation.h"

#include "aclnnop/aclnn_moe_token_unpermute.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;

MoeTokenUnpermuteOperation::MoeTokenUnpermuteOperation(const std::string& name) : AclNnOperation(name) {}

MoeTokenUnpermuteOperation::~MoeTokenUnpermuteOperation() {}

atb::Status MoeTokenUnpermuteOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = NUM2;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(2).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t MoeTokenUnpermuteOperation::GetInputNum() const { return NUM3; }

uint32_t MoeTokenUnpermuteOperation::GetOutputNum() const { return NUM1; }

int MoeTokenUnpermuteOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeTokenUnpermuteGetWorkspaceSize start";

    int ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                     aclInTensors_.at(1).tensor,
                                                     aclInTensors_.at(2).tensor,
                                                     false,
                                                     nullptr,
                                                     aclOutTensors_.at(0).tensor,
                                                     &workspaceSize,
                                                     &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMoeTokenUnpermuteGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int MoeTokenUnpermuteOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " MoeTokenUnpermuteOperation start";
    int ret = aclnnMoeTokenUnpermute(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " MoeTokenUnpermuteOperation end, ret:" << ret;
    return ret;
}

atb::Operation* MoeTokenUnpermuteOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    atb::Operation* op = new MoeTokenUnpermuteOperation(opName);
    return op;
}

REGISTER_OPERATION(MoeTokenUnpermuteOperation, MoeTokenUnpermuteOperationCreate);

}  // namespace dicp
