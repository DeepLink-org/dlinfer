#pragma once

#include <cstdint>
#include <vector>

#include "ops/aclnn_ops/acl_nn_operation.h"

namespace dicp {

class SliceScatterOperation : public AclNnOperation {
public:
    explicit SliceScatterOperation(const std::string& name, std::vector<int64_t> beginVec, std::vector<int64_t> endVec, std::vector<int64_t> stridesVec,
                                   std::vector<int64_t> axesVec);
    ~SliceScatterOperation() override;

    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
    mutable std::vector<int64_t> beginVec_, endVec_, stridesVec_, axesVec_;
    mutable aclIntArray *begin_, *end_, *strides_, *axes_;

    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
