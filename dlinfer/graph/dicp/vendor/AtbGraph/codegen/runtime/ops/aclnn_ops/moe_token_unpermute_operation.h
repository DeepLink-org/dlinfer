#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class MoeTokenUnpermuteOperation : public AclNnOperation {
public:
    explicit MoeTokenUnpermuteOperation(const std::string& name);
    ~MoeTokenUnpermuteOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
