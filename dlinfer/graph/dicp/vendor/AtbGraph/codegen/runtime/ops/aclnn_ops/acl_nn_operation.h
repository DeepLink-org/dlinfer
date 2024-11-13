#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <nlohmann/json.hpp>

#include <string>

#include "atb/operation.h"
#include "utils/log.h"

namespace dicp {
constexpr size_t SVECTOR_SIZE = 8;

struct AclNnTensor {
    atb::Tensor atbTensor;
    aclTensor* tensor = nullptr;
    int CreateTensor(const std::string& opName);
    int InitTensor(void* executor, const std::string& opName, const size_t index, bool isInput);
};

class AclNnOperation : public atb::Operation {
public:
    explicit AclNnOperation(const std::string& name);
    ~AclNnOperation() override;
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;

protected:
    aclTensor* CreateAclTensor(const AclNnTensor& aclNnTensor);
    atb::Status UpdateAclTensorDataPtr(const atb::VariantPack& variantPack);
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);
    std::string opName_;
    atb::SVector<AclNnTensor> aclInTensors_;
    atb::SVector<AclNnTensor> aclOutTensors_;
    aclOpExecutor* aclExecutor_ = nullptr;

private:
    virtual int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) = 0;
    virtual int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) = 0;
};
}  // namespace dicp
