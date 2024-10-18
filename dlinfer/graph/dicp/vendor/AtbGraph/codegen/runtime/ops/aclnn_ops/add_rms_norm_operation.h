#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnAddRmsNormOperation : public AclNnOperation {
public:
    explicit AclNnAddRmsNormOperation(const std::string& name, float epsilon);
    ~AclNnAddRmsNormOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    float epsilon = 1e-5;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnAddRmsNormOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float eps = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("epsilon")) {
        eps = paramJson["epsilon"].get<float>();
    }
    DICP_LOG(INFO) << "AclNnAddRmsNormOperation: name: " << opName << " epsilon:" << eps;
    atb::Operation* op = new AclNnAddRmsNormOperation(opName, eps);
    return op;
}

}  // namespace dicp
