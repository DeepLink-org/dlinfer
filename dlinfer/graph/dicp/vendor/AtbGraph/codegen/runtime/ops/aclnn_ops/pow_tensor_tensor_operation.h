#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnPowTensorTensorOperation : public AclNnOperation {
public:
    explicit AclNnPowTensorTensorOperation(const std::string& name);
    ~AclNnPowTensorTensorOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnPowTensorTensorOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float exponent;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnPowTensorTensorOperation: name: " << opName;
    atb::Operation* op = new AclNnPowTensorTensorOperation(opName);
    return op;
}

}  // namespace dicp
