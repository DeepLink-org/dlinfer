#pragma once

#include <unordered_map>
#include <vector>

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class CustomFusedLoraOperation : public atb::Operation {
public:
    explicit CustomFusedLoraOperation(const std::string& name, const std::string& dtype);
    ~CustomFusedLoraOperation() override;

    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    atb::SVector<AclNnTensor> aclInTensors_;
    atb::SVector<AclNnTensor> aclOutTensors_;

    AclNnTensor CreateTensor(const atb::Tensor& atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);
    void ClearAclScalrs();
    void ClearInternal();

    // Helper functions for weight tensor creation and offset calculation
    atb::Tensor CreateWeightTensor(const atb::Tensor& baseTensor, int64_t rank, int64_t dim, uint64_t offset);
    uint64_t CalculateWeightOffset(const std::vector<int32_t>& ranksVec, size_t adapterId, uint64_t tensorSizePerRank);

private:
    std::string opName_;
    std::string dtype_;
    std::vector<aclScalar*> aclScalingScalar_;

    std::vector<atb::Tensor> weightA_;
    std::vector<atb::Tensor> weightB_;
    std::vector<atb::Tensor> weightATranspose_;

    std::vector<AclNnTensor> aclWeightA_;
    std::vector<AclNnTensor> aclWeightB_;
    std::vector<AclNnTensor> aclWeightATranspose_;

    // adapterId, weightA_index
    std::unordered_map<int32_t, int32_t> weightATransposeIdMap_;
    std::unordered_map<int32_t, aclOpExecutor*> aclWeightAPermuteExecutor_;
    std::unordered_map<int32_t, uint64_t> aclWeightAPermuteWorkspace_;

    uint64_t loraAGroupedGemmWorkspace_ = 0;
    uint64_t loraBGroupedGemmWorkspace_ = 0;

    aclOpExecutor* aclLoraAGroupedGemmExecutor_ = nullptr;
    aclOpExecutor* aclLoraBGroupedGemmExecutor_ = nullptr;

    std::vector<atb::Tensor> scalingWeight_;
    std::vector<atb::Tensor> scalingInput_;
    std::vector<AclNnTensor> aclScalingInput_;
    std::vector<AclNnTensor> aclScalingWeight_;

    std::vector<uint64_t> aclScalingWorkspace_;
    std::vector<aclOpExecutor*> aclScalingExecutor_;
};

}  // namespace dicp
