#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnInplaceDivOperation : public AclNnOperation {
public:
    explicit AclNnInplaceDivOperation(const std::string& name);
    ~AclNnInplaceDivOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnInplaceDivOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float divisor;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnInplaceDivOperation: name: " << opName;
    atb::Operation* op = new AclNnInplaceDivOperation(opName);
    return op;
}

}  // namespace dicp
