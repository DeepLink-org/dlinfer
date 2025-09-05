#include "incre_flash_attention_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_incre_flash_attention_v4.h"
#include "utils/log.h"
#include "utils/tensor_utils.h"

namespace dicp {
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;

AclNnIncreFlashAttentionOperation::AclNnIncreFlashAttentionOperation(const std::string& name, const std::string& inputLayout, float scaleValue)
    : AclNnOperation(name), inputLayout(inputLayout), scaleValue(scaleValue) {}

AclNnIncreFlashAttentionOperation::~AclNnIncreFlashAttentionOperation() {}

atb::Status AclNnIncreFlashAttentionOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                          atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnIncreFlashAttentionOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnIncreFlashAttentionOperation::GetOutputNum() const { return NUM1; }

int AclNnIncreFlashAttentionOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int64_t numHeads = aclInTensors_.at(0).atbTensor.desc.shape.dims[1];
    int64_t numKeyValueHeads = aclInTensors_.at(1).atbTensor.desc.shape.dims[1];
    int64_t sequenceLengthkv = aclInTensors_.at(1).atbTensor.desc.shape.dims[2];

    int kvTensorNum = 1;
    DICP_LOG(INFO) << opName_ << " aclnnIncreFlashAttentionGetWorkspaceSize start";

    std::vector<aclTensor*> tensorsOfKey{aclInTensors_.at(1).tensor};
    tensorKeyList_ = aclCreateTensorList(tensorsOfKey.data(), tensorsOfKey.size());

    std::vector<aclTensor*> tensorsOfValue{aclInTensors_.at(2).tensor};
    tensorValueList_ = aclCreateTensorList(tensorsOfValue.data(), tensorsOfValue.size());

    std::vector<int64_t> actualSeqlenVector = {sequenceLengthkv};
    actualSeqLengths_ = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    char layerOut[this->inputLayout.length()];
    strcpy(layerOut, this->inputLayout.c_str());

    int64_t blockSize = 0;
    int64_t innerPrecise = 1;

    int ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(aclInTensors_.at(0).tensor,
                                                         tensorKeyList_,
                                                         tensorValueList_,
                                                         nullptr,
                                                         nullptr,
                                                         actualSeqLengths_,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         numHeads,
                                                         this->scaleValue,
                                                         layerOut,
                                                         numKeyValueHeads,
                                                         blockSize,
                                                         innerPrecise,
                                                         aclOutTensors_.at(0).tensor,
                                                         &workspaceSize,
                                                         &aclExecutor_);

    DICP_LOG(INFO) << opName_ << "  test aclnnIncreFlashAttentionGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnIncreFlashAttentionOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnIncreFlashAttention start";
    int ret = aclnnIncreFlashAttentionV4(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnIncreFlashAttention end, ret:" << ret;

    if (actualSeqLengths_ != nullptr) {
        aclDestroyIntArray(actualSeqLengths_);
        actualSeqLengths_ = nullptr;
    }
    if (tensorKeyList_ != nullptr) {
        aclDestroyTensorList(tensorKeyList_);
        tensorKeyList_ = nullptr;
    }
    if (tensorValueList_ != nullptr) {
        aclDestroyTensorList(tensorValueList_);
        tensorValueList_ = nullptr;
    }
    aclInTensors_.at(1).tensor = nullptr;
    aclInTensors_.at(2).tensor = nullptr;

    return ret;
}

atb::Operation* AclNnIncreFlashAttentionOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::string inputLayout;
    float scaleValue;

    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("input_layout")) {
        inputLayout = paramJson["input_layout"].get<std::string>();
    }
    if (paramJson.contains("scaleValue")) {
        scaleValue = paramJson["scaleValue"].get<float>();
    }

    DICP_LOG(INFO) << "AclNnIncreFlashAttentionOperation: name: " << opName << "inputLayout" << inputLayout << " scaleValue:" << scaleValue;
    atb::Operation* op = new AclNnIncreFlashAttentionOperation(opName, inputLayout, scaleValue);
    return op;
}

REGISTER_OPERATION(AclNnIncreFlashAttentionOperation, AclNnIncreFlashAttentionOperationCreate);

}  // namespace dicp
