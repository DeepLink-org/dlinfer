#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnSoftmaxOperation : public AclNnOperation {
public:
    explicit AclNnSoftmaxOperation(const std::string& name, int64_t dim);
    ~AclNnSoftmaxOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t dim_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnSoftmaxOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("axes")) {
        auto tmp = paramJson["axes"].get<std::vector<int64_t>>();
        dim = tmp[0];
    }
    DICP_LOG(INFO) << "AclNnSoftmaxOperation: name: " << opName << " dim:" << dim;
    atb::Operation* op = new AclNnSoftmaxOperation(opName, dim);
    return op;
}

}  // namespace dicp
