#include "renormalize_operation.h"

#include <cstddef>

#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "ops/operation_creator.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

RenormalizeOperation::RenormalizeOperation(const std::string& name, int64_t dim) : opName_(name), dim_(dim) {}

RenormalizeOperation::~RenormalizeOperation() {
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclDestroyTensor(aclInTensors_[i].tensor);
    }
    aclInTensors_.clear();
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

std::string RenormalizeOperation::GetName() const { return opName_; }

atb::Status RenormalizeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    // reduceSum out
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = i == dim_ ? 1 : inTensorDescs.at(0).shape.dims[i];
    }

    // div out
    outTensorDescs.at(1).format = inTensorDescs.at(0).format;
    outTensorDescs.at(1).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(1).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";

    return 0;
}

uint32_t RenormalizeOperation::GetInputNum() const { return NUM1; }

uint32_t RenormalizeOperation::GetOutputNum() const { return NUM2; }

AclNnTensor RenormalizeOperation::CreateTensor(atb::Tensor atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

int RenormalizeOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
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

int RenormalizeOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " RenormalizeOperationGetWorkspaceSize start";

    if (context == nullptr) {
        DICP_LOG(ERROR) << opName_ << " setup context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    DICP_CHECK_RET(CreateAclTensors(variantPack));

    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclInTensors_.at(i).CreateTensor(opName_);
    }

    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclOutTensors_.at(i).CreateTensor(opName_);
    }

    // reduceSum
    std::vector<int64_t> dims_{-1};
    aclIntArray* reduceDims = aclCreateIntArray(dims_.data(), dims_.size());
    aclDataType reduceSumDtype = aclInTensors_.at(0).atbTensor.desc.dtype;
    DICP_LOG(INFO) << opName_ << " aclnnReduceSumGetWorkspaceSize start";
    int ret = aclnnReduceSumGetWorkspaceSize(
        aclInTensors_.at(0).tensor, reduceDims, true, reduceSumDtype, aclOutTensors_.at(0).tensor, &reduceSumWorkspaceSize_, &aclReduceSumExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnReduceSumGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << reduceSumWorkspaceSize_
                   << ", aclExecutor:" << aclReduceSumExecutor_;

    workspaceSize = reduceSumWorkspaceSize_;

    // div
    DICP_LOG(INFO) << opName_ << " aclnnDivGetWorkspaceSize start";
    ret = aclnnDivGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOutTensors_.at(0).tensor, aclOutTensors_.at(1).tensor, &divWorkspaceSize_, &aclDivExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnDivGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << divWorkspaceSize_ << ", aclExecutor:" << aclDivExecutor_;

    workspaceSize = divWorkspaceSize_ > workspaceSize ? divWorkspaceSize_ : workspaceSize;

    return 0;
}

int RenormalizeOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
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

    // reduceSum
    aclInTensors_.at(0).atbTensor.deviceData = variantPack.inTensors.at(0).deviceData;
    aclOutTensors_.at(0).atbTensor.deviceData = variantPack.outTensors.at(0).deviceData;
    DICP_CHECK_RET(aclInTensors_.at(0).InitTensor(aclReduceSumExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclReduceSumExecutor_, opName_, 0, false));
    DICP_LOG(INFO) << opName_ << " aclnnReduceSum start";
    int ret = aclnnReduceSum(workspace, reduceSumWorkspaceSize_, aclReduceSumExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnReduceSum end, ret:" << ret;

    // div
    aclOutTensors_.at(1).atbTensor.deviceData = variantPack.outTensors.at(1).deviceData;
    DICP_CHECK_RET(aclInTensors_.at(0).InitTensor(aclDivExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclDivExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(1).InitTensor(aclDivExecutor_, opName_, 1, false));
    DICP_LOG(INFO) << opName_ << " aclnnDiv start";
    ret = aclnnDiv(workspace, divWorkspaceSize_, aclDivExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnDiv end, ret:" << ret;
    DICP_LOG(INFO) << opName_ << " execute end" << 0;

    return 0;
}

atb::Operation* RenormalizeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<std::int64_t>();
    }
    DICP_LOG(INFO) << "RenormalizeOperation: name: " << opName << ", dim:" << dim;
    atb::Operation* op = new RenormalizeOperation(opName, dim);
    return op;
}

REGISTER_OPERATION(RenormalizeOperation, RenormalizeOperationCreate);

}  // namespace dicp
