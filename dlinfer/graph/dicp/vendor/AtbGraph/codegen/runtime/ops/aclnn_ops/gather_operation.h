#pragma once

#include <cstdint>

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class AclNnGatherOperation : public AclNnOperation {
public:
    explicit AclNnGatherOperation(const std::string& name, int64_t dim);
    ~AclNnGatherOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnGatherOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnGatherOperation: name: " << opName;
    atb::Operation* op = new AclNnGatherOperation(opName, dim);
    return op;
}

}  // namespace dicp
