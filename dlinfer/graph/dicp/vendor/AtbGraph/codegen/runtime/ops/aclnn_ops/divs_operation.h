#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnDivsOperation : public AclNnOperation {
public:
    explicit AclNnDivsOperation(const std::string& name, float divisor, const std::string& dtype);
    ~AclNnDivsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar divisor_;
    aclScalar* aclDivisor_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
