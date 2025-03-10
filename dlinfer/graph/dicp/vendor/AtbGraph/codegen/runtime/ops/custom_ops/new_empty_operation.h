#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class NewEmptyOperation : public AclNnOperation {
public:
    explicit NewEmptyOperation(const std::string& name, const std::vector<std::string>& size);
    ~NewEmptyOperation() override;
    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    std::string opName_;
    std::vector<int64_t> size_;
    std::unordered_map<int64_t, std::string> dynamic_size_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};
}  // namespace dicp
