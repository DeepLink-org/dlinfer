#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnCastOperation : public AclNnOperation {
public:
    explicit AclNnCastOperation(const std::string& name, aclDataType dtype);
    ~AclNnCastOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    aclDataType dtype_;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnCastOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("outTensorType")) {
        dataType = static_cast<aclDataType>(paramJson["outTensorType"].get<int32_t>());
    }
    DICP_LOG(INFO) << "AclNnCastOperation name: " << opName << " datatype: " << dataType;
    atb::Operation* op = new AclNnCastOperation(opName, dataType);
    return op;
}

}  // namespace dicp
