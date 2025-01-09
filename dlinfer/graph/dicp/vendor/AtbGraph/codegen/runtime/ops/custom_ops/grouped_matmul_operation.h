// #pragma once

// #include <cstdint>
// #include <string>
// #include "ops/aclnn_ops/acl_nn_operation.h"

// namespace dicp {

// class AclNnGroupedMatmulOperation : public atb::Operation {
// public:
//     explicit AclNnGroupedMatmulOperation(const std::string& name, int64_t splitItem);
//     ~AclNnGroupedMatmulOperation() override;

//     std::string GetName() const override;
//     atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
//     atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
//     atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
//     uint32_t GetInputNum() const override;
//     uint32_t GetOutputNum() const override;

// protected:
//     std::string opName_;
//     int64_t splitItem_;

//     aclOpExecutor* aclPermuteExecutor_ = nullptr;
//     aclOpExecutor* aclGroupedMatmulExecutor_ = nullptr;

//     uint64_t permuteWorkspaceSize_ = 0;
//     uint64_t groupedMatmulWorkspaceSize_ = 0;

// private:
//     atb::SVector<AclNnTensor> aclInTensors_;
//     atb::SVector<AclNnTensor> aclOutTensors_;

//     atb::SVector<aclTensorList*> aclInTensorList_;
//     atb::SVector<aclTensorList*> aclOutTensorList_;

//     aclScalar* aclStart_ = nullptr;
//     aclScalar* aclEnd_ = nullptr;
//     aclScalar* aclStep_ = nullptr;

//     AclNnTensor CreateTensor(atb::Tensor atbTensor);
//     int CreateAclTensors(const atb::VariantPack& variantPack);
// };

// }  // namespace dicp
