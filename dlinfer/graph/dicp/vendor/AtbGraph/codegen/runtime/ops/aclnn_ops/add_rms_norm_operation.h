#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnAddRmsNormOperation : public AclNnOperation {
public:
    explicit AclNnAddRmsNormOperation(const std::string& name, float epsilon);
    ~AclNnAddRmsNormOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    float epsilon = 1e-5;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
