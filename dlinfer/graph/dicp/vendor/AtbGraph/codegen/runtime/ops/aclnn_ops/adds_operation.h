#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnAddsOperation : public AclNnOperation {
public:
    explicit AclNnAddsOperation(const std::string& name, float value, float aplpha, const std::string& dtype);
    ~AclNnAddsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar other_;
    DICPScalar alpha_;
    aclScalar* aclOther_ = nullptr;
    aclScalar* aclAlpha_ = nullptr;

    std::string dtype_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
