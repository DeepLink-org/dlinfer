#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnBatchMatMulOperation : public AclNnOperation {
public:
    explicit AclNnBatchMatMulOperation(const std::string& name, int8_t cubeMathType);
    ~AclNnBatchMatMulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int8_t cubeMathType = 1;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
