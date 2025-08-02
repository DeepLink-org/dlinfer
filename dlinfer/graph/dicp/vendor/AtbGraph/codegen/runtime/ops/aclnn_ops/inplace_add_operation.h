#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnInplaceAddOperation : public AclNnOperation {
public:
    explicit AclNnInplaceAddOperation(const std::string& name, float aplpha, const std::string& dtype);
    ~AclNnInplaceAddOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar alpha_;
    aclScalar* aclAlpha_ = nullptr;

    std::string dtype_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
