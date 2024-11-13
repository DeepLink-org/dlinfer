#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnDivsOperation : public AclNnOperation {
public:
    explicit AclNnDivsOperation(const std::string& name, float divisor, const std::string& dtype);
    ~AclNnDivsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar divisor_;
    aclScalar* aclDivisor_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnDivsOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::string dtype;
    float divisor;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("divisor")) {
        divisor = paramJson["divisor"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnDivsOperation: name: " << opName << " divisor:" << divisor;
    atb::Operation* op = new AclNnDivsOperation(opName, divisor, dtype);
    return op;
}

}  // namespace dicp
