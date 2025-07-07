#include "split_with_size_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_split_with_size.h"
#include "utils/common.h"
#include "utils/global_dict.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnSplitWithSizeOperation::AclNnSplitWithSizeOperation(const std::string& name, int64_t splitDim, std::vector<std::string> splitSizes)
    : AclNnOperation(name), splitDim_(splitDim) {
    splitSizes_.resize(splitSizes.size());
    for (size_t i = 0; i < splitSizes_.size(); ++i) {
        bool isDynamic = !std::isdigit(splitSizes[i][0]);
        if (isDynamic) {
            dynamicSplitSizesMap_[i] = splitSizes[i];
        } else {
            splitSizes_[i] = std::stol(splitSizes[i]);
        }
    }
    if (dynamicSplitSizesMap_.size() == 0) {
        aclSplitSizes_ = aclCreateIntArray(splitSizes_.data(), splitSizes_.size());
    }
}

AclNnSplitWithSizeOperation::~AclNnSplitWithSizeOperation() {}

atb::Status AclNnSplitWithSizeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    const auto& inputTensorDesc = inTensorDescs.at(0);
    const auto inputDimNum = inputTensorDesc.shape.dimNum;
    const auto inputFormat = inputTensorDesc.format;
    const auto inputDtype = inputTensorDesc.dtype;

    const auto& inputDims = inputTensorDesc.shape.dims;
    int64_t splitDim = splitDim_ >= 0 ? splitDim_ : inputDimNum + splitDim_;

    auto& globalDict = GetGlobalDictData();
    for (size_t i = 0; i < splitSizes_.size(); ++i) {
        auto& outputTensorDesc = outTensorDescs.at(i);
        outputTensorDesc.format = inputFormat;
        outputTensorDesc.shape.dimNum = inputDimNum;
        outputTensorDesc.dtype = inputDtype;
        auto& outputDims = outputTensorDesc.shape.dims;

        int64_t targetDimValue = -1;
        auto dynamicSize = dynamicSplitSizesMap_.find(i);
        if (dynamicSize != dynamicSplitSizesMap_.end()) {
            auto it = globalDict.find(dynamicSize->second);
            if (it != globalDict.end()) {
                targetDimValue = static_cast<int64_t>(it->second);
            } else {
                DICP_LOG(ERROR) << "Cannot find key " << dynamicSize->second << "  in global_dict";
            }
        } else {
            targetDimValue = splitSizes_[i];
        }

        for (size_t j = 0; j < inputDimNum; ++j) {
            outputDims[j] = (j != splitDim) ? inputDims[j] : targetDimValue;
        }
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnSplitWithSizeOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnSplitWithSizeOperation::GetOutputNum() const { return splitSizes_.size(); }

int AclNnSplitWithSizeOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnSplitWithSizeGetWorkspaceSize start";
    std::vector<aclTensor*> tmp;
    tmp.resize(aclOutTensors_.size());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        tmp[i] = aclOutTensors_.at(i).tensor;
    }
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

    if (dynamicSplitSizesMap_.size() > 0) {
        auto& globalDict = GetGlobalDictData();
        for (auto& [key, value] : dynamicSplitSizesMap_) {
            auto it = globalDict.find(value);
            if (it != globalDict.end()) {
                splitSizes_[key] = static_cast<int64_t>(it->second);
            } else {
                DICP_LOG(ERROR) << "Cannot find key " << value << " in global dict";
            }
        }
        if (aclSplitSizes_ != nullptr) {
            aclDestroyIntArray(aclSplitSizes_);
            aclSplitSizes_ = nullptr;
        }
        aclSplitSizes_ = aclCreateIntArray(splitSizes_.data(), splitSizes_.size());
    }
    int ret = aclnnSplitWithSizeGetWorkspaceSize(aclInTensors_.at(0).tensor, aclSplitSizes_, splitDim_, tensorList, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnSplitWithSizeGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSplitWithSizeOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnSplitWithSize start";
    int ret = aclnnSplitWithSize(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnSplitWithSize end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnSplitWithSizeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t splitDim;
    std::vector<std::string> splitSizes;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("splitDim")) {
        splitDim = paramJson["splitDim"].get<int64_t>();
    }
    if (paramJson.contains("splitSizes")) {
        splitSizes = paramJson["splitSizes"].get<std::vector<std::string>>();
    }
    DICP_LOG(INFO) << "AclNnSplitWithSizeOperation: name: " << opName;
    atb::Operation* op = new AclNnSplitWithSizeOperation(opName, splitDim, splitSizes);
    return op;
}

REGISTER_OPERATION(AclNnSplitWithSizeOperation, AclNnSplitWithSizeOperationCreate);

}  // namespace dicp
