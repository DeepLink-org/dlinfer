#pragma once

#include <vector>

#include "acl_nn_operation.h"

namespace dicp {
class AclNnInplaceIndexCopyOperation : public AclNnOperation {
public:
    explicit AclNnInplaceIndexCopyOperation(const std::string& name, int64_t dim);
    ~AclNnInplaceIndexCopyOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    mutable std::vector<int64_t> indexVec_;
    mutable aclTensor* index_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
