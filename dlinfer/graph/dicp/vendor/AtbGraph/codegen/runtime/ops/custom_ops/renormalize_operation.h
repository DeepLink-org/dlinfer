#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class RenormalizeOperation : public atb::Operation {
public:
    explicit RenormalizeOperation(const std::string& name, int64_t dim);
    ~RenormalizeOperation() override;

    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
    int64_t dim_;

    aclOpExecutor* aclReduceSumExecutor_ = nullptr;
    aclOpExecutor* aclDivExecutor_ = nullptr;

    uint64_t reduceSumWorkspaceSize_ = 0;
    uint64_t divWorkspaceSize_ = 0;

    aclIntArray* reduceDims_ = nullptr;

private:
    atb::SVector<AclNnTensor> aclInTensors_;
    atb::SVector<AclNnTensor> aclOutTensors_;

    AclNnTensor CreateTensor(atb::Tensor atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);
};

}  // namespace dicp
