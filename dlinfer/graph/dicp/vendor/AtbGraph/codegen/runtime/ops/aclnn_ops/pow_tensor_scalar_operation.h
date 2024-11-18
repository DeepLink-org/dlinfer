#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnPowTensorScalarOperation : public AclNnOperation {
public:
    explicit AclNnPowTensorScalarOperation(const std::string& name, float exponent, const std::string& dtype);
    ~AclNnPowTensorScalarOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar exponent_;
    aclScalar* aclExponent_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnPowTensorScalarOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float exponent;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("exponent")) {
        exponent = paramJson["exponent"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnPowTensorScalarOperation: name: " << opName << " exponent:" << exponent << " dtype:" << dtype;
    atb::Operation* op = new AclNnPowTensorScalarOperation(opName, exponent, dtype);
    return op;
}

}  // namespace dicp
