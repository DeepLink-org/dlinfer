#pragma once
#include <vector>

#include "acl_nn_operation.h"

namespace dicp {
class AclNnSplitWithSizeOperation : public AclNnOperation {
public:
    explicit AclNnSplitWithSizeOperation(const std::string& name, int64_t splitDim, std::vector<int64_t> splitSizes);
    ~AclNnSplitWithSizeOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t splitDim_;
    std::vector<int64_t> splitSizes_;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnSplitWithSizeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t splitDim;
    std::vector<int64_t> splitSizes;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("splitDim")) {
        splitDim = paramJson["splitDim"].get<int64_t>();
    }
    if (paramJson.contains("splitSizes")) {
        splitSizes = paramJson["splitSizes"].get<std::vector<int64_t>>();
    }
    DICP_LOG(INFO) << "AclNnSplitWithSizeOperation: name: " << opName;
    atb::Operation* op = new AclNnSplitWithSizeOperation(opName, splitDim, splitSizes);
    return op;
}

}  // namespace dicp
