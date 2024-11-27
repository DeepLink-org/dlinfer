#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnSubOperation : public AclNnOperation {
public:
    explicit AclNnSubOperation(const std::string& name, float aplpha, const std::string& dtype);
    ~AclNnSubOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar alpha_;
    aclScalar* aclAlpha_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnSubOperationCreate(const nlohmann::json& paramJson) {
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
    DICP_LOG(INFO) << "AclNnSubOperation: name: " << opName << " alpha:" << alpha << " dtype:" << dtype;
    atb::Operation* op = new AclNnSubOperation(opName, alpha, dtype);
    return op;
}

}  // namespace dicp
