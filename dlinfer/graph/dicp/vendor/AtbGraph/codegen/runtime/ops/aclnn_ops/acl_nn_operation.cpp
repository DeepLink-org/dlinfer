#include "acl_nn_operation.h"

#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

int AclNnTensor::CreateTensor(const std::string& opName) {
    atb::SVector<int64_t> strides(atbTensor.desc.shape.dimNum, 1);
    for (int64_t i = atbTensor.desc.shape.dimNum - 2; i >= 0; i--) {
        strides[i] = atbTensor.desc.shape.dims[i + 1] * strides[i + 1];
    }

    tensor = aclCreateTensor(atbTensor.desc.shape.dims,
                             atbTensor.desc.shape.dimNum,
                             atbTensor.desc.dtype,
                             strides.data(),
                             0,
                             atbTensor.desc.format,
                             atbTensor.desc.shape.dims,
                             atbTensor.desc.shape.dimNum,
                             atbTensor.deviceData);
    if (tensor) {
        return atb::NO_ERROR;
    }

    DICP_LOG(ERROR) << opName << " aclCreateTensor fail";
    return atb::ERROR_INTERNAL_ERROR;
}

int AclNnTensor::InitTensor(void* executor, const std::string& opName, const size_t index, bool isInput) {
    int ret = 0;
    if (isInput) {
        ret = AclSetInputTensorAddr(static_cast<aclOpExecutor*>(executor), index, tensor, atbTensor.deviceData);
    } else {
        ret = AclSetOutputTensorAddr(static_cast<aclOpExecutor*>(executor), index, tensor, atbTensor.deviceData);
    }

    DICP_LOG_IF(ret != 0, ERROR) << opName << " aclInitTensor fail, error:" << ret;
    return ret;
}

AclNnOperation::AclNnOperation(const std::string& opName) : opName_(opName) {}

AclNnOperation::~AclNnOperation() {
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclDestroyTensor(aclInTensors_[i].tensor);
    }
    aclInTensors_.clear();

    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

std::string AclNnOperation::GetName() const { return opName_; }

static const uint64_t ACTIVATION_INDEX = 0;
static const uint64_t BIAS_INDEX = 4;

atb::Status AclNnOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " setup start";

    if (context == nullptr) {
        DICP_LOG(ERROR) << opName_ << " setup context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    DICP_CHECK_RET(CreateAclTensors(variantPack));

    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        DICP_CHECK_RET(aclInTensors_.at(i).CreateTensor(opName_));
    }

    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        auto ret = aclOutTensors_.at(i).CreateTensor(opName_);
        if (ret != 0) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }

    DICP_CHECK_RET(SetAclNnWorkspaceExecutor(workspaceSize));

    return atb::NO_ERROR;
}

int AclNnOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
    DICP_LOG(INFO) << opName_ << " CreateAclTensor start";
    aclInTensors_.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclInTensors_[i] = CreateTensor(variantPack.inTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " Create aclInTensor end";

    aclOutTensors_.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " Create aclOutTensor end";
    DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

atb::Status AclNnOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
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

    DICP_CHECK_RET(UpdateAclTensorDataPtr(variantPack));
    DICP_CHECK_RET(CallAclExecute(workspace, workspaceSize, aclExecutor_, stream));

    DICP_LOG(INFO) << opName_ << " execute end";

    return atb::NO_ERROR;
}

atb::Status AclNnOperation::UpdateAclTensorDataPtr(const atb::VariantPack& variantPack) {
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        AclNnTensor& aclNnTensor = aclInTensors_[i];
        aclNnTensor.atbTensor.deviceData = variantPack.inTensors.at(i).deviceData;
        DICP_CHECK_RET(aclNnTensor.InitTensor(aclExecutor_, opName_, i, true));
    }

    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        AclNnTensor& aclNnTensor = aclOutTensors_[i];
        aclNnTensor.atbTensor.deviceData = variantPack.outTensors.at(i).deviceData;
        DICP_CHECK_RET(aclNnTensor.InitTensor(aclExecutor_, opName_, i, false));
    }

    return atb::NO_ERROR;
}

AclNnTensor AclNnOperation::CreateTensor(atb::Tensor atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

}  // namespace dicp
