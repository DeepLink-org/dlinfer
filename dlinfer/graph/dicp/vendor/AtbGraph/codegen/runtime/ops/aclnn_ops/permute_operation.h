#pragma once
#include <vector>

#include "acl_nn_operation.h"

namespace dicp {
class AclNnPermuteOperation : public AclNnOperation {
public:
    explicit AclNnPermuteOperation(const std::string& name, std::vector<int64_t> dims);
    ~AclNnPermuteOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    std::vector<int64_t> dims_;
    int CreateAclTensors(const atb::VariantPack& variantPack) override;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};

inline atb::Operation* AclNnPermuteOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> dims;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("perm")) {
        dims = paramJson["perm"].get<std::vector<int64_t>>();
    }
    DICP_LOG(INFO) << "AclNnPermuteOperation: name: " << opName;
    atb::Operation* op = new AclNnPermuteOperation(opName, dims);
    return op;
}

}  // namespace dicp
