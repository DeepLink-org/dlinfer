#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnDivOperation : public AclNnOperation {
public:
    explicit AclNnDivOperation(const std::string& name);
    ~AclNnDivOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
