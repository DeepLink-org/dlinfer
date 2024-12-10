#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnTopkOperation : public AclNnOperation {
public:
    explicit AclNnTopkOperation(const std::string& name, int64_t k, int64_t dim);
    ~AclNnTopkOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t k_;
    int64_t dim_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnTopkOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t k;
    int64_t dim = -1;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("num")) {
        k = paramJson["num"].get<int64_t>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnTopkOperation: name: " << opName << " k:" << k << " dim:" << dim;
    atb::Operation* op = new AclNnTopkOperation(opName, k, dim);
    return op;
}

}  // namespace dicp
