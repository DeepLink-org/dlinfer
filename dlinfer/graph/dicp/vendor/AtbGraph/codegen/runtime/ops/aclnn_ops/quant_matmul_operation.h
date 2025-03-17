#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnQuantMatmulOperation : public AclNnOperation {
public:
    explicit AclNnQuantMatmulOperation(const std::string& name);
    ~AclNnQuantMatmulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
