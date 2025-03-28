#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class AclNnMoeGatingTopkSoftmaxOperation : public AclNnOperation {
public:
    explicit AclNnMoeGatingTopkSoftmaxOperation(const std::string& name, int64_t topk, int64_t renorm, bool outputSoftmaxResultFlag);
    ~AclNnMoeGatingTopkSoftmaxOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t topk_;
    int64_t renorm_;
    bool outputSoftmaxResultFlag_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
