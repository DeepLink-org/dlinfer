#include "masked_fill_scalar_operation.h"

#include <algorithm>

#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "aclnnop/aclnn_mul.h"
#include "utils/common.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

std::string MaskedFillScalarOperation::GetName() const { return opName_; }

MaskedFillScalarOperation::MaskedFillScalarOperation(const std::string& name, float value, const std::string& dtype) : opName_(name) {
    value_ = DICPScalar(value, dtype);
    aclValue_ = aclCreateScalar(value_.getValuePtr(), value_.getDataType());
    one_ = DICPScalar(1.0, dtype);
    aclOne_ = aclCreateScalar(one_.getValuePtr(), one_.getDataType());
}

MaskedFillScalarOperation::~MaskedFillScalarOperation() {
    if (aclValue_ != nullptr) {
        aclDestroyScalar(aclValue_);
    }
    if (aclOne_ != nullptr) {
        aclDestroyScalar(aclOne_);
    }
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

atb::Status MaskedFillScalarOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t MaskedFillScalarOperation::GetInputNum() const { return NUM2; }

uint32_t MaskedFillScalarOperation::GetOutputNum() const { return NUM1; }

atb::Status MaskedFillScalarOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " setup start";

    if (context == nullptr) {
        DICP_LOG(ERROR) << opName_ << " setup context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    DICP_CHECK_RET(CreateAclTensors(variantPack));

    DICP_CHECK_RET(aclInTensors_.at(0).CreateTensor(opName_));
    DICP_CHECK_RET(aclInTensors_.at(1).CreateTensor(opName_));
    DICP_CHECK_RET(aclOutTensors_.at(0).CreateTensor(opName_));

    int ret = aclnnMulsGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOne_, aclOutTensors_.at(0).tensor, &mulsWorkspaceSize_, &aclMulsExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << mulsWorkspaceSize_
                   << ", aclExecutor:" << aclMulsExecutor_;

    ret = aclnnInplaceMaskedFillScalarGetWorkspaceSize(
        aclOutTensors_.at(0).tensor, aclInTensors_.at(1).tensor, aclValue_, &inplaceMaskedFillScalarWorkspaceSize_, &aclInplaceMaskedFillScalarExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMaskedFillScalarGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << inplaceMaskedFillScalarWorkspaceSize_
                   << ", aclExecutor:" << aclInplaceMaskedFillScalarExecutor_;

    workspaceSize = std::max(mulsWorkspaceSize_, inplaceMaskedFillScalarWorkspaceSize_);

    return atb::NO_ERROR;
}

atb::Status MaskedFillScalarOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
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
    aclInTensors_.at(0).atbTensor.deviceData = variantPack.inTensors.at(0).deviceData;
    aclInTensors_.at(1).atbTensor.deviceData = variantPack.inTensors.at(1).deviceData;
    aclOutTensors_.at(0).atbTensor.deviceData = variantPack.outTensors.at(0).deviceData;
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclMulsExecutor_, opName_, 0, false));
    DICP_CHECK_RET(aclInTensors_.at(0).InitTensor(aclMulsExecutor_, opName_, 0, true));

    DICP_LOG(INFO) << opName_ << " aclnnMuls start";
    int ret = aclnnMuls(workspace, mulsWorkspaceSize_, aclMulsExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMuls end, ret:" << ret;

    // inplace masked fill scalar
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclInplaceMaskedFillScalarExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclInTensors_.at(1).InitTensor(aclInplaceMaskedFillScalarExecutor_, opName_, 0, true));

    DICP_LOG(INFO) << opName_ << " aclnnAdds start";
    ret = aclnnInplaceMaskedFillScalar(workspace, inplaceMaskedFillScalarWorkspaceSize_, aclInplaceMaskedFillScalarExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceMaskedFillScalar end, ret:" << ret;
    DICP_LOG(INFO) << opName_ << " execute end";

    return atb::NO_ERROR;
}

int MaskedFillScalarOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
    DICP_LOG(INFO) << opName_ << " CreateAclTensor start";

    aclInTensors_.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclInTensors_[i] = CreateTensor(variantPack.inTensors.at(i));
    }

    aclOutTensors_.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " Create aclOutTensor end";
    DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

AclNnTensor MaskedFillScalarOperation::CreateTensor(atb::Tensor atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

atb::Operation* MaskedFillScalarOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "MaskedFillScalarOperation: name: " << opName << ", value: " << value << ", dtype: " << dtype;
    atb::Operation* op = new MaskedFillScalarOperation(opName, value, dtype);
    return op;
}

REGISTER_OPERATION(MaskedFillScalarOperation, MaskedFillScalarOperationCreate);

}  // namespace dicp
