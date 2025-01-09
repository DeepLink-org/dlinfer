#include "moe_token_permute_operation.h"

#include "aclnnop/aclnn_moe_token_permute.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM7 = 7;

MoeTokenPermuteOperation::MoeTokenPermuteOperation(const std::string& name) : AclNnOperation(name) {}

MoeTokenPermuteOperation::~MoeTokenPermuteOperation() {}

atb::Status MoeTokenPermuteOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    auto seq_len = inTensorDescs.at(1).shape.dims[0];
    auto topk = inTensorDescs.at(1).shape.dims[1];

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dims[0] = seq_len * topk;
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

    outTensorDescs.at(1).format = inTensorDescs.at(1).format;
    outTensorDescs.at(1).shape.dimNum = NUM1;
    outTensorDescs.at(1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(1).shape.dims[0] = seq_len * topk;
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t MoeTokenPermuteOperation::GetInputNum() const { return NUM2; }

uint32_t MoeTokenPermuteOperation::GetOutputNum() const { return NUM2; }

int MoeTokenPermuteOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeTokenPermuteGetWorkspaceSize start";
    DICP_LOG(INFO) << opName_ << "aclInTensors_.size: " << aclInTensors_.size() << " aclOutTensors_.size:" << aclOutTensors_.size();

    int ret = aclnnMoeTokenPermuteGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                   aclInTensors_.at(1).tensor,
                                                   0,
                                                   false,
                                                   aclOutTensors_.at(0).tensor,
                                                   aclOutTensors_.at(1).tensor,
                                                   &workspaceSize,
                                                   &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMoeTokenPermuteGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int MoeTokenPermuteOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " MoeTokenPermuteOperation start";
    int ret = aclnnMoeTokenPermute(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " MoeTokenPermuteOperation end, ret:" << ret;
    return ret;
}

atb::Operation* MoeTokenPermuteOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    atb::Operation* op = new MoeTokenPermuteOperation(opName);
    return op;
}

REGISTER_OPERATION(MoeTokenPermuteOperation, MoeTokenPermuteOperationCreate);

}  // namespace dicp
