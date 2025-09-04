#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnCatOperation : public AclNnOperation {
public:
    explicit AclNnCatOperation(const std::string& name, int32_t inputNum, int32_t concatDim);
    ~AclNnCatOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int32_t concatDim = -1;
    int32_t inputNum = -1;
    aclTensorList* tensorList_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
