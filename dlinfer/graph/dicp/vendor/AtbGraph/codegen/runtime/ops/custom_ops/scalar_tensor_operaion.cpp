#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "ops/operation_creator.h"
#include "scalar_tensor_operation.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;

std::string ScalarTensorOperation::GetName() const { return opName_; }

ScalarTensorOperation::ScalarTensorOperation(const std::string& name, float value, const std::string& dtype) : opName_(name) {
    value_ = DICPScalar(value, dtype);
    zero_ = DICPScalar(0, dtype);
    alpha_ = DICPScalar(1, dtype);
    aclValue_ = aclCreateScalar(value_.getValuePtr(), value_.getDataType());
    aclZero_ = aclCreateScalar(zero_.getValuePtr(), zero_.getDataType());
    aclAlpha_ = aclCreateScalar(alpha_.getValuePtr(), alpha_.getDataType());
}

ScalarTensorOperation::~ScalarTensorOperation() {
    if (aclValue_ != nullptr) {
        aclDestroyScalar(aclValue_);
    }
    if (aclZero_ != nullptr) {
        aclDestroyScalar(aclZero_);
    }
    if (aclAlpha_ != nullptr) {
        aclDestroyScalar(aclAlpha_);
    }
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

atb::Status ScalarTensorOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(0).shape.dimNum = 1;
    outTensorDescs.at(0).shape.dims[0] = 1;
    outTensorDescs.at(0).dtype = value_.getDataType();
    return 0;
}

uint32_t ScalarTensorOperation::GetInputNum() const { return NUM0; }

uint32_t ScalarTensorOperation::GetOutputNum() const { return NUM1; }

atb::Status ScalarTensorOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " setup start";

    if (context == nullptr) {
        DICP_LOG(ERROR) << opName_ << " setup context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    ClearAclTensors();
    DICP_CHECK_RET(CreateAclTensors(variantPack));

    int ret = aclOutTensors_.at(0).CreateTensor(opName_);
    if (ret != 0) {
        return atb::ERROR_INTERNAL_ERROR;
    }

    ret = aclnnInplaceMulsGetWorkspaceSize(aclOutTensors_.at(0).tensor, aclZero_, &mulsWorkspaceSize_, &aclZeroExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << mulsWorkspaceSize_
                   << ", aclExecutor:" << aclZeroExecutor_;

    ret = aclnnAddsGetWorkspaceSize(aclOutTensors_.at(0).tensor, aclValue_, aclAlpha_, aclOutTensors_.at(0).tensor, &addsWorkspaceSize_, &aclAddsExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << addsWorkspaceSize_
                   << ", aclExecutor:" << aclAddsExecutor_;

    return atb::NO_ERROR;
}

atb::Status ScalarTensorOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " execute start";
    if (!context) {
        DICP_LOG(ERROR) << opName_ << " execute fail, context param is null";
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        DICP_LOG(ERROR) << opName_ << " execute fail, execute stream in context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    // mul
    aclOutTensors_.at(0).atbTensor.deviceData = variantPack.outTensors.at(0).deviceData;
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclZeroExecutor_, opName_, 0, true));
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMuls start";
    int ret = aclnnInplaceMuls(workspace, mulsWorkspaceSize_, aclZeroExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMuls end, ret:" << ret;

    // add
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclAddsExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclAddsExecutor_, opName_, 0, false));

    DICP_LOG(INFO) << opName_ << " aclnnAdds start";
    ret = aclnnAdds(workspace, addsWorkspaceSize_, aclAddsExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnAdds end, ret:" << ret;
    DICP_LOG(INFO) << opName_ << " execute end";

    return atb::NO_ERROR;
}

int ScalarTensorOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
    DICP_LOG(INFO) << opName_ << " CreateAclTensor start";

    aclOutTensors_.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " Create aclOutTensor end";
    DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

void ScalarTensorOperation::ClearAclTensors() {
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        if (aclOutTensors_[i].tensor != nullptr) {
            aclDestroyTensor(aclOutTensors_[i].tensor);
            aclOutTensors_[i].tensor = nullptr;
        }
    }
}

AclNnTensor ScalarTensorOperation::CreateTensor(atb::Tensor atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

atb::Operation* CustomScalarTensorOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value") && paramJson.contains("valueStr")) {
        std::string valueStr = paramJson["valueStr"].get<std::string>();
        if (valueStr == "") {
            value = paramJson["value"].get<float>();
        } else {
            if (valueStr == "inf") {
                value = std::numeric_limits<float>::infinity();
            } else if (valueStr == "-inf") {
                value = -std::numeric_limits<float>::infinity();
            } else {
                throw std::runtime_error("invalid valueStr: " + valueStr);
            }
        }
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "CustomScalarTensorOperation: name: " << opName << " value:" << value << " dtype:" << dtype;
    atb::Operation* op = new ScalarTensorOperation(opName, value, dtype);
    return op;
}

REGISTER_OPERATION(CustomScalarTensorOperation, CustomScalarTensorOperationCreate);

}  // namespace dicp
