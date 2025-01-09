#pragma once

#include <cstdint>
#include <string>

#include "acl_nn_operation.h"

namespace dicp {

class AclNnGroupedMatmulOperation : public AclNnOperation {
public:
    explicit AclNnGroupedMatmulOperation(const std::string& name, int64_t splitItem);
    ~AclNnGroupedMatmulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t splitItem = 2;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
