
#include "cat_operation.h"

#include <aclnn/acl_meta.h>
#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"
#include "utils/log.h"

namespace dicp {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

AclNnCatOperation::AclNnCatOperation(const std::string& name, int32_t inputNum, int32_t concatDim) : AclNnOperation(name) {
    this->concatDim = concatDim;
    this->inputNum = inputNum;
}

AclNnCatOperation::~AclNnCatOperation() {}

atb::Status AclNnCatOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    int64_t concatDimSize = 0;
    for (size_t i = 0; i < inTensorDescs.size(); ++i) {
        concatDimSize += inTensorDescs.at(i).shape.dims[this->concatDim];
    }
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        if (static_cast<int32_t>(i) == this->concatDim) {
            outTensorDescs.at(0).shape.dims[i] = concatDimSize;
        } else {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        }
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnCatOperation::GetInputNum() const { return this->inputNum; }

uint32_t AclNnCatOperation::GetOutputNum() const { return NUM1; }

int AclNnCatOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnCatGetWorkspaceSize start";
    std::vector<aclTensor*> tmp;
    tmp.resize(this->inputNum);
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        tmp[i] = aclInTensors_.at(i).tensor;
    }
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    int ret = aclnnCatGetWorkspaceSize(tensorList, this->concatDim, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnCatGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnCatOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnCat start";
    int ret = aclnnCat(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnCat end, ret:" << ret;
    return ret;
}

}  // namespace dicp
