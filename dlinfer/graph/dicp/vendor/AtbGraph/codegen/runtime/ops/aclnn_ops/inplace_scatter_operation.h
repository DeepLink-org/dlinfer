#pragma once

#include <cstdint>

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnInplaceScatterOperation : public AclNnOperation {
public:
    explicit AclNnInplaceScatterOperation(const std::string& name, int64_t dim, int64_t reduceType);
    ~AclNnInplaceScatterOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    int64_t reduceType_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
