#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ops/aclnn_ops/acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class ScalarTensorOperation : public atb::Operation {
public:
    explicit ScalarTensorOperation(const std::string& name, float value, const std::string& dtype);
    ~ScalarTensorOperation();
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;

private:
    aclTensor* CreateAclTensor(const AclNnTensor& aclNnTensor);
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);

private:
    std::string opName_;
    DICPScalar value_;
    DICPScalar zero_;
    DICPScalar alpha_;
    aclScalar* aclValue_ = nullptr;
    aclScalar* aclZero_ = nullptr;
    aclScalar* aclAlpha_ = nullptr;
    aclOpExecutor* aclZeroExecutor_ = nullptr;
    aclOpExecutor* aclAddsExecutor_ = nullptr;
    uint64_t mulsWorkspaceSize_ = 0;
    uint64_t addsWorkspaceSize_ = 0;
    atb::SVector<AclNnTensor> aclOutTensors_;
};

inline atb::Operation* CustomScalarTensorOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("value") && paramJson.contains("valueStr")) {
        std::string valueStr = paramJson["valueStr"].get<std::string>();
        if (valueStr == "") {
            value = paramJson["value"].get<float>();
        } else {
            if (valueStr == "inf") {
                value = std::numeric_limits<float>::infinity();
            } else if (valueStr == "-inf") {
                value = -std::numeric_limits<float>::infinity();
            } else {
                throw std::runtime_error("invalid valueStr: " + valueStr);
            }
        }
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "CustomScalarTensorOperation: name: " << opName << " value:" << value << " dtype:" << dtype;
    atb::Operation* op = new ScalarTensorOperation(opName, value, dtype);
    return op;
}

}  // namespace dicp
