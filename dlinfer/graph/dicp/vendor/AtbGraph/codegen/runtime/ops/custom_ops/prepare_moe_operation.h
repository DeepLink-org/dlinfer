#pragma once

#include <cstdint>

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class PrepareMoeOperation : public atb::Operation {
public:
    explicit PrepareMoeOperation(const std::string& name, int64_t numExperts);
    ~PrepareMoeOperation() override;

    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
    int64_t numExperts_;
    int64_t topk_;
    int64_t seqLength_;

    aclOpExecutor* aclArangeExecutor_ = nullptr;
    aclOpExecutor* aclPermuteExecutor_ = nullptr;
    aclOpExecutor* aclBincountExecutor_ = nullptr;
    aclOpExecutor* aclCumsumExecutor_ = nullptr;

    uint64_t arangeWorkspaceSize_ = 0;
    uint64_t permuteWorkspaceSize_ = 0;
    uint64_t bincountWorkspaceSize_ = 0;
    uint64_t cumsumWorkspaceSize_ = 0;

private:
    atb::SVector<AclNnTensor> aclInTensors_;
    atb::SVector<AclNnTensor> aclOutTensors_;

    aclScalar* aclStart_ = nullptr;
    aclScalar* aclEnd_ = nullptr;
    aclScalar* aclStep_ = nullptr;

    AclNnTensor CreateTensor(atb::Tensor atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);
};

}  // namespace dicp
