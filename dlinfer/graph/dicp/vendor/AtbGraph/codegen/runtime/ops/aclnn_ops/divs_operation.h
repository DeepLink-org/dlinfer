#pragma once

#include "acl_nn_operation.h"

namespace dicp {
class AclNnDivsOperation : public AclNnOperation {
public:
    explicit AclNnDivsOperation(const std::string& name, float divisor);
    ~AclNnDivsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    aclScalar* divisor_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnDivsOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float divisor;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("divisor")) {
        divisor = paramJson["divisor"].get<float>();
    }
    DICP_LOG(INFO) << "AclNnDivsOperation: name: " << opName << " divisor:" << divisor;
    atb::Operation* op = new AclNnDivsOperation(opName, divisor);
    return op;
}

}  // namespace dicp
