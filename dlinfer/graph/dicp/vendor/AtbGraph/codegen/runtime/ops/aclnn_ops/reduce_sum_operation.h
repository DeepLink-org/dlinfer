#pragma once

#include <vector>

#include "acl/acl.h"
#include "acl_nn_operation.h"
#include "utils/scalar.h"
namespace dicp {

class AclNnReduceSumOperation : public AclNnOperation {
public:
    explicit AclNnReduceSumOperation(const std::string& name, const std::vector<int64_t>& dims, bool keepDim, const std::string& dtype);
    ~AclNnReduceSumOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;

private:
    std::vector<int64_t> dims_;
    aclIntArray* aclDims_ = nullptr;
    bool keepDim_;
    aclDataType dtype_;
};

}  // namespace dicp
