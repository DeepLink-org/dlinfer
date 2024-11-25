#pragma once
#include <vector>

#include "acl_nn_operation.h"

namespace dicp {
class AclNnPermuteOperation : public AclNnOperation {
public:
    explicit AclNnPermuteOperation(const std::string& name, std::vector<int64_t> dims);
    ~AclNnPermuteOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    std::vector<int64_t> dims_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
