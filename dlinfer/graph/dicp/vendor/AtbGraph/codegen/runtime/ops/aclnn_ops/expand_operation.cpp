#include "expand_operation.h"

#include <aclnn/acl_meta.h>

#include <algorithm>

#include "acl/acl.h"
#include "aclnnop/aclnn_expand.h"
#include "utils/log.h"

namespace dicp {
const int NUM1 = 1;

AclNnExpandOperation::AclNnExpandOperation(const std::string& name, std::vector<int64_t> size)
    : AclNnOperation(name), size_(std::move(size)), needUpdateSize_(false) {
    needUpdateSize_ = std::any_of(size_.begin(), size_.end(), [](int64_t val) { return val == -1; });

    if (!needUpdateSize_) {
        aclSize_ = aclCreateIntArray(size_.data(), size_.size());
    }
}

AclNnExpandOperation::~AclNnExpandOperation() {
    if (aclSize_) {
        aclDestroyIntArray(aclSize_);
    }
}

atb::Status AclNnExpandOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < size_.size(); ++i) {
        outTensorDescs.at(0).shape.dims[i] = (size_[i] == -1) ? inTensorDescs.at(0).shape.dims[i] : size_[i];
    }
    return 0;
}

uint32_t AclNnExpandOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnExpandOperation::GetOutputNum() const { return NUM1; }

int AclNnExpandOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    if (needUpdateSize_) {
        std::vector<int64_t> newSize{size_.begin(), size_.end()};
        for (size_t i = 0; i < size_.size(); ++i) {
            if (size_[i] == -1) {
                newSize[i] = aclInTensors_.at(0).atbTensor.desc.shape.dims[i];
            }
        }
        if (aclSize_ != nullptr) {
            aclDestroyIntArray(aclSize_);
            aclSize_ = nullptr;
        }
        aclSize_ = aclCreateIntArray(newSize.data(), newSize.size());
    }

    return aclnnExpandGetWorkspaceSize(aclInTensors_.at(0).tensor, aclSize_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
}

int AclNnExpandOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnExpand(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnExpand end, ret:" << ret;
    return ret;
}

}  // namespace dicp
