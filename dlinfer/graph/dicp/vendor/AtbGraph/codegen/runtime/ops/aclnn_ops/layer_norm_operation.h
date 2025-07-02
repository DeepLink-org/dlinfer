#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnLayerNormOperation : public AclNnOperation {
public:
    explicit AclNnLayerNormOperation(const std::string& name, float epsilon, std::vector<int64_t>& normDim);
    ~AclNnLayerNormOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    float epsilon = 1e-5;
    std::vector<int64_t> normDim;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
