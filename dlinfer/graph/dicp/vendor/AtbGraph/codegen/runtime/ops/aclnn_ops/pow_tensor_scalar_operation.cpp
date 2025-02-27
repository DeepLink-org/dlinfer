#include "pow_tensor_scalar_operation.h"

#include "aclnnop/aclnn_pow.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnPowTensorScalarOperation::AclNnPowTensorScalarOperation(const std::string& name, float exponent, const std::string& dtype) : AclNnOperation(name) {
    exponent_ = DICPScalar(exponent, dtype);
    aclExponent_ = aclCreateScalar(exponent_.getValuePtr(), exponent_.getDataType());
}

AclNnPowTensorScalarOperation::~AclNnPowTensorScalarOperation() {
    if (aclExponent_ != nullptr) {
        aclDestroyScalar(aclExponent_);
    }
}

atb::Status AclNnPowTensorScalarOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnPowTensorScalarOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnPowTensorScalarOperation::GetOutputNum() const { return NUM1; }

int AclNnPowTensorScalarOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorScalarGetWorkspaceSize start";

    int ret = aclnnPowTensorScalarGetWorkspaceSize(aclInTensors_.at(0).tensor, aclExponent_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorScalarGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnPowTensorScalarOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorScalar start";
    int ret = aclnnPowTensorScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnPowTensorScalar end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnPowTensorScalarOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float exponent;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("exponent")) {
        exponent = paramJson["exponent"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnPowTensorScalarOperation: name: " << opName << " exponent:" << exponent << " dtype:" << dtype;
    atb::Operation* op = new AclNnPowTensorScalarOperation(opName, exponent, dtype);
    return op;
}

REGISTER_OPERATION(AclNnPowTensorScalarOperation, AclNnPowTensorScalarOperationCreate);

}  // namespace dicp
