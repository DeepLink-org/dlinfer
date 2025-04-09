#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class MaskedFillScalarOperation : public AclNnOperation {
public:
    explicit MaskedFillScalarOperation(const std::string& name, float value, const std::string& dtype);
    ~MaskedFillScalarOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;

private:
    DICPScalar value_;
    aclScalar* aclValue_ = nullptr;
    DICPScalar one_;
    aclScalar* aclOne_ = nullptr;
    aclOpExecutor* aclMulsExecutor_ = nullptr;
    uint64_t mulsWorkspaceSize_ = 0;
};

}  // namespace dicp
