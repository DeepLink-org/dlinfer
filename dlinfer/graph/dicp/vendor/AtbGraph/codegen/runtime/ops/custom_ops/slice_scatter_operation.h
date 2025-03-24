#pragma once

#include <cstdint>
#include <vector>

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class SliceScatterOperation : public AclNnOperation {
public:
    explicit SliceScatterOperation(const std::string& name, int64_t dim, int64_t start, int64_t end, int64_t step);
    ~SliceScatterOperation() override;

    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
    int64_t dim_, start_, end_, step_;
    mutable std::vector<int64_t> beginVec_, endVec_, stridesVec_, axesVec_;
    mutable aclIntArray *beginArray_, *endArray_, *stridesArray_, *axesArray_;

    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
