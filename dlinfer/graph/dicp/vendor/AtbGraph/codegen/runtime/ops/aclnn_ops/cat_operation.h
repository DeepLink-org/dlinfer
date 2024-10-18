#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnCatOperation : public AclNnOperation {
public:
    explicit AclNnCatOperation(const std::string& name, int32_t inputNum, int32_t concatDim);
    ~AclNnCatOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int32_t concatDim = -1;
    int32_t inputNum = -1;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnCatOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int32_t inputNum = 0;
    int32_t concatDim = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("inputNum")) {
        inputNum = paramJson["inputNum"].get<int32_t>();
    }
    if (paramJson.contains("concatDim")) {
        concatDim = paramJson["concatDim"].get<int32_t>();
    }

    DICP_LOG(INFO) << "AclNnCatOperation: name: " << opName << " inputNum:" << inputNum << " concatDim:" << concatDim;
    atb::Operation* op = new AclNnCatOperation(opName, inputNum, concatDim);
    return op;
}

}  // namespace dicp
