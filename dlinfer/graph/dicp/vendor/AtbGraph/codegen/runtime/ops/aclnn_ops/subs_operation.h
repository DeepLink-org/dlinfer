#pragma once

#include "acl_nn_operation.h"
#include "scalar.h"

namespace dicp {

class AclNnSubsOperation : public AclNnOperation {
public:
    explicit AclNnSubsOperation(const std::string& name, float value, float aplpha, const std::string& dtype);
    ~AclNnSubsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar other_;
    DICPScalar alpha_;
    aclScalar* aclOther_ = nullptr;
    aclScalar* aclAlpha_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnSubsOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    float alpha;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<float>();
    }
    if (paramJson.contains("alpha")) {
        alpha = paramJson["alpha"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnSubsOperation: name: " << opName << " value:" << value << " alpha:" << alpha << " dtype:" << dtype;
    atb::Operation* op = new AclNnSubsOperation(opName, value, alpha, dtype);
    return op;
}

}  // namespace dicp
