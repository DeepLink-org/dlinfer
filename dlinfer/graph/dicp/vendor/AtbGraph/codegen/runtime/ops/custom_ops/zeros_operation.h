#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class ZerosOperation : public AclNnOperation {
public:
    explicit ZerosOperation(const std::string& name, const std::vector<int64_t>& size, aclDataType dtype);
    ~ZerosOperation() override;

    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
    std::vector<int64_t> size_;
    aclDataType dtype_;

    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
