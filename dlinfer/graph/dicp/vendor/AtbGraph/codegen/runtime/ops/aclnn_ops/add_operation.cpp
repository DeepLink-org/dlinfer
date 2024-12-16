#include "add_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_add.h"
#include "utils/log.h"
#include "utils/misc.h"
#include "utils/scalar.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnAddOperation::AclNnAddOperation(const std::string& name, float alpha, const std::string& dtype) : AclNnOperation(name) {
    alpha_ = DICPScalar(alpha, dtype);
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

AclNnAddOperation::~AclNnAddOperation() {
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
}

atb::Status AclNnAddOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = std::max(inTensorDescs.at(0).shape.dimNum, inTensorDescs.at(1).shape.dimNum);

    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        auto dim0 = i < inTensorDescs.at(0).shape.dimNum ? inTensorDescs.at(0).shape.dims[i] : -1;
        auto dim1 = i < inTensorDescs.at(1).shape.dimNum ? inTensorDescs.at(1).shape.dims[i] : -1;
        outTensorDescs.at(0).shape.dims[i] = std::max({dim0, dim1});
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnAddOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnAddOperation::GetOutputNum() const { return NUM1; }

int AclNnAddOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " AclNnAddGetWorkspaceSize start";
    int ret =
        aclnnAddGetWorkspaceSize(aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclAlpha_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " AclNnAddGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnAddOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " AclNnAdd start";
    int ret = aclnnAdd(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " AclNnAdd end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnAddOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float alpha;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("alpha")) {
        alpha = paramJson["alpha"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnAddOperation: name: " << opName << " alpha:" << alpha << " dtype:" << dtype;
    atb::Operation* op = new AclNnAddOperation(opName, alpha, dtype);
    return op;
}

REGISTER_OPERATION(AclNnAddOperation, AclNnAddOperationCreate);

}  // namespace dicp
