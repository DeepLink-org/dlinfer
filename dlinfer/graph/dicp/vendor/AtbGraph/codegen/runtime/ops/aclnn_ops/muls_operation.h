#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnMulsOperation : public AclNnOperation {
public:
    explicit AclNnMulsOperation(const std::string& name, float value, const std::string& dtype);
    ~AclNnMulsOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar other_;
    aclScalar* aclOther_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnMulsOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnMulsOperation: name: " << opName << " value:" << value << " dtype:" << dtype;
    atb::Operation* op = new AclNnMulsOperation(opName, value, dtype);
    return op;
}

}  // namespace dicp
