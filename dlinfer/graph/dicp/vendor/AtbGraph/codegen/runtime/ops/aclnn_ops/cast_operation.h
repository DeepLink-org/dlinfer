#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnCastOperation : public AclNnOperation {
public:
    explicit AclNnCastOperation(const std::string& name, aclDataType dtype);
    ~AclNnCastOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    aclDataType dtype_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
