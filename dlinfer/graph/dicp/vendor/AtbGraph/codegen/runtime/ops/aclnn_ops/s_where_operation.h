#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnSWhereOperation : public AclNnOperation {
public:
    explicit AclNnSWhereOperation(const std::string& name);
    ~AclNnSWhereOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnSWhereOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnSWhereOperation: name: " << opName;
    atb::Operation* op = new AclNnSWhereOperation(opName);
    return op;
}

}  // namespace dicp
