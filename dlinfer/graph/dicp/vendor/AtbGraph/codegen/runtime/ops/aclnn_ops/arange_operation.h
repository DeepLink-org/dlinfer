#pragma once

#include "acl_nn_operation.h"

namespace dicp {

class AclNnArangeOperation : public AclNnOperation {
public:
    explicit AclNnArangeOperation(const std::string& name, int64_t start, int64_t end, int64_t step);
    ~AclNnArangeOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t start_;
    int64_t end_;
    int64_t step_;
    int64_t sizeArange_;
    aclScalar* aclStart_ = nullptr;
    aclScalar* aclEnd_ = nullptr;
    aclScalar* aclStep_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnArangeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t start = 0;
    int64_t end = 0;
    int64_t step = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("start")) {
        start = paramJson["start"].get<int64_t>();
    }
    if (paramJson.contains("end")) {
        end = paramJson["end"].get<int64_t>();
    }
    if (paramJson.contains("step")) {
        step = paramJson["step"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnArangeOperation: name: " << opName << " start:" << start << " end:" << end << " step:" << step;
    atb::Operation* op = new AclNnArangeOperation(opName, start, end, step);
    return op;
}

}  // namespace dicp
