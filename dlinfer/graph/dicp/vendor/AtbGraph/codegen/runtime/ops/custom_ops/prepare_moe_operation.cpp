#include "prepare_moe_operation.h"

#include <cstddef>

#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_bincount.h"
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_permute.h"
#include "nlohmann/json.hpp"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM4 = 4;

PrepareMoeOperation::PrepareMoeOperation(const std::string& name, int64_t numExperts) : opName_(name), numExperts_(numExperts) {}

PrepareMoeOperation::~PrepareMoeOperation() {
    if (aclStart_ != nullptr) {
        aclDestroyScalar(aclStart_);
    }
    if (aclEnd_ != nullptr) {
        aclDestroyScalar(aclEnd_);
    }
    if (aclStep_ != nullptr) {
        aclDestroyScalar(aclStep_);
    }

    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclDestroyTensor(aclInTensors_[i].tensor);
    }
    aclInTensors_.clear();
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclDestroyTensor(aclOutTensors_[i].tensor);
    }
    aclOutTensors_.clear();
}

std::string PrepareMoeOperation::GetName() const { return opName_; }

atb::Status PrepareMoeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    // arange out
    outTensorDescs.at(0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(0).shape.dimNum = 2;
    outTensorDescs.at(0).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];

    // permute out
    outTensorDescs.at(1).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(1).shape.dimNum = 2;
    outTensorDescs.at(1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

    // bincount out
    outTensorDescs.at(2).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(2).shape.dimNum = 1;
    outTensorDescs.at(2).dtype = aclDataType::ACL_INT64;
    outTensorDescs.at(2).shape.dims[0] = numExperts_;

    // cumsum out
    outTensorDescs.at(3).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(3).shape.dimNum = 1;
    outTensorDescs.at(3).dtype = aclDataType::ACL_INT64;
    outTensorDescs.at(3).shape.dims[0] = numExperts_;

    DICP_LOG(INFO) << opName_ << " infer shape end";

    return 0;
}

uint32_t PrepareMoeOperation::GetInputNum() const { return NUM1; }

uint32_t PrepareMoeOperation::GetOutputNum() const { return NUM4; }

AclNnTensor PrepareMoeOperation::CreateTensor(atb::Tensor atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

int PrepareMoeOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
    DICP_LOG(INFO) << opName_ << " CreateAclTensor start";

    aclInTensors_.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        aclInTensors_[i] = CreateTensor(variantPack.inTensors.at(i));
    }
    auto seqLength = aclInTensors_[0].atbTensor.desc.shape.dims[0];
    auto topk = aclInTensors_[0].atbTensor.desc.shape.dims[1];
    aclInTensors_[0].atbTensor.desc.shape.dimNum = 1;
    aclInTensors_[0].atbTensor.desc.shape.dims[0] = seqLength * topk;

    aclOutTensors_.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

int PrepareMoeOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " PrepareMoeOperationGetWorkspaceSize start";

    seqLength_ = variantPack.inTensors[0].desc.shape.dims[0];
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

    // arange
    DICP_LOG(INFO) << opName_ << " aclnnArangeGetWorkspaceSize start";
    auto seqLength = variantPack.inTensors.at(0).desc.shape.dims[0];
    auto topk = variantPack.inTensors.at(0).desc.shape.dims[1];
    int64_t startValue = 0, endValue = seqLength * topk, stepValue = 1;
    aclStart_ = aclCreateScalar(&startValue, aclDataType::ACL_INT64);
    aclEnd_ = aclCreateScalar(&endValue, aclDataType::ACL_INT64);
    aclStep_ = aclCreateScalar(&stepValue, aclDataType::ACL_INT64);
    int ret = aclnnArangeGetWorkspaceSize(aclStart_, aclEnd_, aclStep_, aclOutTensors_.at(0).tensor, &arangeWorkspaceSize_, &aclArangeExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnArangeGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << arangeWorkspaceSize_
                   << ", aclExecutor:" << aclArangeExecutor_;

    workspaceSize = arangeWorkspaceSize_;

    // permute
    DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize start";
    std::vector<int64_t> permuteDimsValue{1, 0};
    aclIntArray* permuteDims = aclCreateIntArray(permuteDimsValue.data(), permuteDimsValue.size());
    ret = aclnnPermuteGetWorkspaceSize(aclOutTensors_.at(0).tensor, permuteDims, aclOutTensors_.at(1).tensor, &permuteWorkspaceSize_, &aclPermuteExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << permuteWorkspaceSize_
                   << ", aclExecutor:" << aclPermuteExecutor_;

    workspaceSize = permuteWorkspaceSize_ > workspaceSize ? permuteWorkspaceSize_ : workspaceSize;

    // bincount
    DICP_LOG(INFO) << opName_ << " aclnnBincountGetWorkspaceSize start";
    ret = aclnnBincountGetWorkspaceSize(
        aclInTensors_.at(0).tensor, nullptr, numExperts_, aclOutTensors_.at(2).tensor, &bincountWorkspaceSize_, &aclBincountExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnBincountGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << bincountWorkspaceSize_
                   << ", aclExecutor:" << aclBincountExecutor_;

    workspaceSize = bincountWorkspaceSize_ > workspaceSize ? bincountWorkspaceSize_ : workspaceSize;

    // cumsum
    DICP_LOG(INFO) << opName_ << " aclnnCumsumGetWorkspaceSize start";
    ret = aclnnCumsumGetWorkspaceSize(
        aclOutTensors_.at(2).tensor, 0, aclDataType::ACL_INT64, aclOutTensors_.at(3).tensor, &cumsumWorkspaceSize_, &aclCumsumExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnCumsumGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << cumsumWorkspaceSize_
                   << ", aclExecutor:" << aclCumsumExecutor_;

    workspaceSize = cumsumWorkspaceSize_ > workspaceSize ? cumsumWorkspaceSize_ : workspaceSize;

    return 0;
}

int PrepareMoeOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
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

    // arange
    aclOutTensors_.at(0).atbTensor.deviceData = variantPack.outTensors.at(0).deviceData;
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclArangeExecutor_, opName_, 0, false));
    DICP_LOG(INFO) << opName_ << " aclnnArange start";
    int ret = aclnnArange(workspace, arangeWorkspaceSize_, aclArangeExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnArange end, ret:" << ret;

    // permute
    aclOutTensors_.at(1).atbTensor.deviceData = variantPack.outTensors.at(1).deviceData;
    DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclPermuteExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(1).InitTensor(aclPermuteExecutor_, opName_, 1, false));
    DICP_LOG(INFO) << opName_ << " aclnnPermute start";
    ret = aclnnPermute(workspace, permuteWorkspaceSize_, aclPermuteExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnPermute end, ret:" << ret;

    // bincount
    aclInTensors_.at(0).atbTensor.deviceData = variantPack.inTensors.at(0).deviceData;
    aclOutTensors_.at(2).atbTensor.deviceData = variantPack.outTensors.at(2).deviceData;
    DICP_CHECK_RET(aclInTensors_.at(0).InitTensor(aclPermuteExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(2).InitTensor(aclPermuteExecutor_, opName_, 1, false));
    DICP_LOG(INFO) << opName_ << " aclnnBincount start";
    ret = aclnnBincount(workspace, bincountWorkspaceSize_, aclBincountExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnBincount end, ret:" << ret;

    // cumsum
    DICP_CHECK_RET(aclOutTensors_.at(2).InitTensor(aclPermuteExecutor_, opName_, 0, true));
    DICP_CHECK_RET(aclOutTensors_.at(3).InitTensor(aclPermuteExecutor_, opName_, 1, false));
    DICP_LOG(INFO) << opName_ << " aclnnCumsum start";
    ret = aclnnCumsum(workspace, cumsumWorkspaceSize_, aclCumsumExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " aclnnCumsum end, ret:" << ret;

    DICP_LOG(INFO) << opName_ << " execute end" << 0;

    return 0;
}

atb::Operation* PrepareMoeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t numExperts;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("numExperts")) {
        numExperts = paramJson["numExperts"].get<std::int64_t>();
    }
    atb::Operation* op = new PrepareMoeOperation(opName, numExperts);
    return op;
}

REGISTER_OPERATION(PrepareMoeOperation, PrepareMoeOperationCreate);

}  // namespace dicp
