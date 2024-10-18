#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnBatchMatMulOperation : public AclNnOperation {
public:
    explicit AclNnBatchMatMulOperation(const std::string& name, int8_t cubeMathType);
    ~AclNnBatchMatMulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int8_t cubeMathType = 1;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnBatchMatMulOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int8_t cubeMathType = 1;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("cubeMathType")) {
        auto tmp = paramJson["cubeMathType"].get<int32_t>();
        cubeMathType = static_cast<int8_t>(tmp);
    }

    DICP_LOG(INFO) << "AclNnBatchMatMulOperation: name: " << opName << " cubeMathType:" << cubeMathType;
    atb::Operation* op = new AclNnBatchMatMulOperation(opName, cubeMathType);
    return op;
}

}  // namespace dicp
