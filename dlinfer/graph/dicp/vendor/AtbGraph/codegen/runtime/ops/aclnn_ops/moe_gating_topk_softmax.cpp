#include "moe_gating_topk_softmax.h"

#include <cstddef>

#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;

AclNnMoeGatingTopkSoftmaxOperation::AclNnMoeGatingTopkSoftmaxOperation(const std::string& name, int64_t topk, int64_t renorm, bool outputSoftmaxResultFlag)
    : AclNnOperation(name), topk_(topk), renorm_(renorm), outputSoftmaxResultFlag_(outputSoftmaxResultFlag) {}

AclNnMoeGatingTopkSoftmaxOperation::~AclNnMoeGatingTopkSoftmaxOperation() {}

atb::Status AclNnMoeGatingTopkSoftmaxOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                           atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = i == outTensorDescs.at(0).shape.dimNum - 1 ? topk_ : inTensorDescs.at(0).shape.dims[i];
    }

    outTensorDescs.at(1).format = outTensorDescs.at(0).format;
    outTensorDescs.at(1).shape.dimNum = outTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(1).dtype = aclDataType::ACL_INT32;
    for (size_t i = 0; i < outTensorDescs.at(1).shape.dimNum; ++i) {
        outTensorDescs.at(1).shape.dims[i] = outTensorDescs.at(0).shape.dims[i];
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnMoeGatingTopkSoftmaxOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnMoeGatingTopkSoftmaxOperation::GetOutputNum() const { return NUM2; }

int AclNnMoeGatingTopkSoftmaxOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize start";

    int ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                          nullptr,
                                                          topk_,
                                                          renorm_,
                                                          outputSoftmaxResultFlag_,
                                                          aclOutTensors_.at(0).tensor,
                                                          aclOutTensors_.at(1).tensor,
                                                          nullptr,
                                                          &workspaceSize,
                                                          &aclExecutor_);

    DICP_LOG(INFO) << opName_ << " aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnMoeGatingTopkSoftmaxOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMoeGatingTopKSoftmaxV2 start";
    int ret = aclnnMoeGatingTopKSoftmaxV2(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMoeGatingTopKSoftmaxV2 end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnMoeGatingTopkSoftmaxOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t topk, renorm;
    bool outputSoftmaxResultFlag;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("topk")) {
        topk = paramJson["topk"].get<int64_t>();
    }
    if (paramJson.contains("renorm")) {
        renorm = paramJson["renorm"].get<int64_t>();
    }
    if (paramJson.contains("outputSoftmaxResultFlag")) {
        outputSoftmaxResultFlag = paramJson["outputSoftmaxResultFlag"].get<bool>();
    }
    DICP_LOG(INFO) << "AclNnMoeGatingTopkSoftmaxOperation: name: " << opName << " topk:" << topk << " renorm:" << renorm
                   << " outputSoftmaxResultFlag:" << outputSoftmaxResultFlag;
    atb::Operation* op = new AclNnMoeGatingTopkSoftmaxOperation(opName, topk, renorm, outputSoftmaxResultFlag);
    return op;
}

REGISTER_OPERATION(AclNnMoeGatingTopkSoftmaxOperation, AclNnMoeGatingTopkSoftmaxOperationCreate);

}  // namespace dicp
