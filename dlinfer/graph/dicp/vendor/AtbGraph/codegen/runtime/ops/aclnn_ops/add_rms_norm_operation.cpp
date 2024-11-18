#include "add_rms_norm_operation.h"

#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm.h"
#include "utils/common.h"
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

AclNnAddRmsNormOperation::AclNnAddRmsNormOperation(const std::string& name, float epsilon) : AclNnOperation(name) { this->epsilon = epsilon; }

AclNnAddRmsNormOperation::~AclNnAddRmsNormOperation() {}

atb::Status AclNnAddRmsNormOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == NUM1) {
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
        if (inTensorDescs.at(0).shape.dimNum == DIM3) {
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            outTensorDescs.at(i).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM2];
        } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
        } else {
            DICP_LOG(ERROR) << opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum;
        }
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnAddRmsNormOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnAddRmsNormOperation::GetOutputNum() const { return NUM3; }

int AclNnAddRmsNormOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnAddRmsNormGetWorkspaceSize start";
    int ret = aclnnAddRmsNormGetWorkspaceSize(aclInTensors_.at(0).tensor,
                                              aclInTensors_.at(1).tensor,
                                              aclInTensors_.at(2).tensor,
                                              this->epsilon,
                                              aclOutTensors_.at(0).tensor,
                                              aclOutTensors_.at(1).tensor,
                                              aclOutTensors_.at(2).tensor,
                                              &workspaceSize,
                                              &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnAddRmsNormGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnAddRmsNormOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnAddRmsNorm start";
    int ret = aclnnAddRmsNorm(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnAddRmsNorm end, ret:" << ret;
    return ret;
}

}  // namespace dicp
