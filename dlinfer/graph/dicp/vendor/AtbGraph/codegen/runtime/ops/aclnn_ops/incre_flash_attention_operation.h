#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnIncreFlashAttentionOperation : public AclNnOperation {
public:
    explicit AclNnIncreFlashAttentionOperation(const std::string& name, const std::string& inputLayout, float scaleValue);
    ~AclNnIncreFlashAttentionOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    std::string inputLayout;
    float scaleValue;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
