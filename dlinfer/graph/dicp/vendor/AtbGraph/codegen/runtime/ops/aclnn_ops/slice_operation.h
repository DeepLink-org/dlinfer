#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnSliceOperation : public AclNnOperation {
public:
    explicit AclNnSliceOperation(const std::string& name, int64_t dim, int64_t start, int64_t end, int64_t step);
    ~AclNnSliceOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
