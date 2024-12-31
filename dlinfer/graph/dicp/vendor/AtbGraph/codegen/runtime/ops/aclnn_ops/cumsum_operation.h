#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnCumsumOperation : public AclNnOperation {
public:
    explicit AclNnCumsumOperation(const std::string& name, int64_t dim, aclDataType dtype);
    ~AclNnCumsumOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    aclDataType dtype_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
