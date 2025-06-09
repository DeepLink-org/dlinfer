#include "layer_norm_operation.h"

#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;

AclNnLayerNormOperation::AclNnLayerNormOperation(const std::string& name, float epsilon, std::vector<int64_t>& normDim)
    : AclNnOperation(name), epsilon(epsilon), normDim(normDim) {}

AclNnLayerNormOperation::~AclNnLayerNormOperation() {}

atb::Status AclNnLayerNormOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    for (size_t i = 0; i < 2; ++i) {
        outTensorDescs.at(i).format = inTensorDescs.at(i).format;
        outTensorDescs.at(i).dtype = inTensorDescs.at(i).dtype;
        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(i).shape.dimNum;
    }

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    for (size_t i = 0; i < outTensorDescs.at(1).shape.dimNum; ++i) {
        outTensorDescs.at(1).shape.dims[i] = 1;
    }
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];

    for (size_t i = 0; i < outTensorDescs.at(2).shape.dimNum; ++i) {
        outTensorDescs.at(2).shape.dims[i] = 1;
    }
    outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnLayerNormOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnLayerNormOperation::GetOutputNum() const { return NUM3; }

int AclNnLayerNormOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnLayerNormGetWorkspaceSize start";
    aclIntArray* normalizedShape = aclCreateIntArray(this->normDim.data(), this->normDim.size());
    int ret = aclnnLayerNormGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                             normalizedShape,
                                             aclInTensors_.at(1).tensor,
                                             aclInTensors_.at(2).tensor,
                                             this->epsilon,
                                             aclOutTensors_.at(0).tensor,
                                             aclOutTensors_.at(1).tensor,
                                             aclOutTensors_.at(2).tensor,
                                             &workspaceSize,
                                             &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnLayerNormGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnLayerNormOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnLayerNorm start";
    int ret = aclnnLayerNorm(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnLayerNorm end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnLayerNormOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float eps;
    std::vector<int64_t> normDim;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("epsilon")) {
        eps = paramJson["epsilon"].get<float>();
    }
    if (paramJson.contains("normDim")) {
        normDim = paramJson["normDim"].get<std::vector<int64_t>>();
    }
    DICP_LOG(INFO) << "AclNnLayerNormOperation: name: " << opName << " epsilon:" << eps;
    atb::Operation* op = new AclNnLayerNormOperation(opName, eps, normDim);
    return op;
}

REGISTER_OPERATION(AclNnLayerNormOperation, AclNnLayerNormOperationCreate);

}  // namespace dicp
