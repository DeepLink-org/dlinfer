#pragma once
#include <cstdint>
#include <vector>

#include "acl_nn_operation.h"

namespace dicp {
class AclNnExpandOperation : public AclNnOperation {
public:
    explicit AclNnExpandOperation(const std::string& name, std::vector<int64_t> size);
    ~AclNnExpandOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    std::vector<int64_t> size_;
    aclIntArray* aclSize_ = nullptr;
    bool needUpdateSize_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
