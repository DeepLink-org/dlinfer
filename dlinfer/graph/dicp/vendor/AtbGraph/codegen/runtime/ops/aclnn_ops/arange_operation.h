#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnArangeOperation : public AclNnOperation {
public:
    explicit AclNnArangeOperation(const std::string& name, int64_t start, int64_t end, int64_t step);
    ~AclNnArangeOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t start_;
    int64_t end_;
    int64_t step_;
    int64_t sizeArange_;
    aclScalar* aclStart_ = nullptr;
    aclScalar* aclEnd_ = nullptr;
    aclScalar* aclStep_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
