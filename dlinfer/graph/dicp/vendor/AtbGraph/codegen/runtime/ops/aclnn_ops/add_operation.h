#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnAddOperation : public AclNnOperation {
public:
    explicit AclNnAddOperation(const std::string& name, float aplpha, const std::string& dtype);
    ~AclNnAddOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar alpha_;
    aclScalar* aclAlpha_ = nullptr;

    std::string dtype_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnAddOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float alpha;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("alpha")) {
        alpha = paramJson["alpha"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnAddOperation: name: " << opName << " alpha:" << alpha << " dtype:" << dtype;
    atb::Operation* op = new AclNnAddOperation(opName, alpha, dtype);
    return op;
}

}  // namespace dicp
