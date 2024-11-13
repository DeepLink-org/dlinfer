#include "utils/workspace.h"

#include <acl/acl.h>

#include <cstdlib>

#include "utils/config.h"
#include "utils/log.h"
#include "utils/tensor_utils.h"

namespace dicp {

constexpr int KB_1 = 1024;
constexpr int MB_1 = 1024 * 1024;
constexpr int GB_1 = 1024 * 1024 * 1024;

Workspace::Workspace() {
    bufferSize_ = GetConfig().WorkspaceBufferSize();

    DICP_LOG(INFO) << "Workspace init, bufferSize:" << bufferSize_;
    if (bufferSize_ > 0) {
        atTensor_ = CreateAtTensor(bufferSize_);
        buffer_ = atTensor_.data_ptr();
    }
}

void* Workspace::GetWorkspaceBuffer(uint64_t bufferSize) {
    if (bufferSize <= bufferSize_) {
        DICP_LOG(INFO) << "GetWorkspaceBuffer bufferSize:" << bufferSize << "<= bufferSize_:" << bufferSize_;
        return atTensor_.data_ptr();
    }

    if (aclrtSynchronizeDevice() != 0) {
        return nullptr;
    }

    atTensor_.reset();
    atTensor_ = CreateAtTensor(bufferSize);
    bufferSize_ = atTensor_.numel();
    DICP_LOG(INFO) << "Workspace new bufferSize:" << bufferSize;
    buffer_ = atTensor_.data_ptr();
    return atTensor_.data_ptr();
}

torch::Tensor Workspace::CreateAtTensor(uint64_t bufferSize) {
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    tensorDesc.shape.dimNum = 2;
    tensorDesc.shape.dims[0] = KB_1;
    tensorDesc.shape.dims[1] = (bufferSize + KB_1 - 1) / KB_1;

    return tensor_utils::CreateAtTensorFromTensorDesc(tensorDesc);
}

void* GetWorkspaceBuffer(uint64_t bufferSize) {
    static Workspace workspace;
    return workspace.GetWorkspaceBuffer(bufferSize);
}

}  // namespace dicp
