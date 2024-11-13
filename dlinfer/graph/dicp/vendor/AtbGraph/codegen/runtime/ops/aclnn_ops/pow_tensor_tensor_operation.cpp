#include "pow_tensor_tensor_operation.h"

#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnPowTensorTensorOperation::AclNnPowTensorTensorOperation(const std::string& name) : AclNnOperation(name) {}

AclNnPowTensorTensorOperation::~AclNnPowTensorTensorOperation() {}

atb::Status AclNnPowTensorTensorOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    auto firstDimNum = inTensorDescs.at(0).shape.dimNum;
    auto secondDimNum = inTensorDescs.at(1).shape.dimNum;
    auto maxDimNum = firstDimNum > secondDimNum ? firstDimNum : secondDimNum;
    outTensorDescs.at(0).shape.dimNum = maxDimNum;
    for (size_t i = 0; i < maxDimNum; ++i) {
        if (i == firstDimNum) {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(1).shape.dims[i];
        }
        if (i == secondDimNum) {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        }
        outTensorDescs.at(0).shape.dims[i] =
            inTensorDescs.at(0).shape.dims[i] > inTensorDescs.at(1).shape.dims[i] ? inTensorDescs.at(0).shape.dims[i] : inTensorDescs.at(1).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnPowTensorTensorOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnPowTensorTensorOperation::GetOutputNum() const { return NUM1; }

int AclNnPowTensorTensorOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorTensorGetWorkspaceSize start";

    int ret = aclnnPowTensorTensorGetWorkspaceSize(
        aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorTensorGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnPowTensorTensorOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorTensor start";
    int ret = aclnnPowTensorTensor(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorTensor end, ret:" << ret;
    return ret;
}

}  // namespace dicp
