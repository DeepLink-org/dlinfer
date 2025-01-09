#include "zeros_like_operation.h"

#include <cstdint>

#include "aclnnop/aclnn_zero.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;

ZerosLikeOperation::ZerosLikeOperation(const std::string& name) : AclNnOperation(name), opName_(name) {}

ZerosLikeOperation::~ZerosLikeOperation() {
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclDestroyTensor(aclInTensors_[i].tensor);
    }
    aclInTensors_.clear();
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

std::string ZerosLikeOperation::GetName() const { return opName_; }

atb::Status ZerosLikeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t ZerosLikeOperation::GetInputNum() const { return NUM1; }

uint32_t ZerosLikeOperation::GetOutputNum() const { return NUM1; }

int ZerosLikeOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZeroGetWorkspaceSize start";

    int ret = aclnnInplaceZeroGetWorkspaceSize(aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZeroGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int ZerosLikeOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZero start";
    int ret = aclnnInplaceZero(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZero end, ret:" << ret;
    return ret;
}

atb::Operation* ZerosLikeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> size;
    aclDataType dtype = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "ZerosLikeOperationCreate: name: " << opName;
    atb::Operation* op = new ZerosLikeOperation(opName);
    return op;
}

REGISTER_OPERATION(ZerosLikeOperation, ZerosLikeOperationCreate);

}  // namespace dicp
