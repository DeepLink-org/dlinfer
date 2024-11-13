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
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

AclNnSplitWithSizeOperation::AclNnSplitWithSizeOperation(const std::string& name, int64_t splitDim, std::vector<int64_t> splitSizes)
    : AclNnOperation(name), splitDim_(splitDim), splitSizes_(std::move(splitSizes)) {}

AclNnSplitWithSizeOperation::~AclNnSplitWithSizeOperation() {}

atb::Status AclNnSplitWithSizeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    const auto& inputTensorDesc = inTensorDescs.at(0);
    const auto inputDimNum = inputTensorDesc.shape.dimNum;
    const auto inputFormat = inputTensorDesc.format;
    const auto inputDtype = inputTensorDesc.dtype;

    for (size_t i = 0; i < splitSizes_.size(); ++i) {
        auto& outputTensorDesc = outTensorDescs.at(i);
        outputTensorDesc.format = inputFormat;
        outputTensorDesc.shape.dimNum = inputDimNum;
        outputTensorDesc.dtype = inputDtype;

        auto& outputDims = outputTensorDesc.shape.dims;
        const auto& inputDims = inputTensorDesc.shape.dims;

        for (size_t j = 0; j < inputDimNum; ++j) {
            outputDims[j] = (j != splitDim_) ? inputDims[j] : splitSizes_[i];
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
    aclIntArray* sizes = aclCreateIntArray(splitSizes_.data(), splitSizes_.size());
    int ret = aclnnSplitWithSizeGetWorkspaceSize(aclInTensors_.at(0).tensor, sizes, splitDim_, tensorList, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnSplitWithSizeGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnSplitWithSizeOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnPermute start";
    int ret = aclnnSplitWithSize(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnPermute end, ret:" << ret;
    return ret;
}

}  // namespace dicp
