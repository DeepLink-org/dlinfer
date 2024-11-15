#pragma once

#include <bits/stdint-intn.h>

#include "atb/operation.h"

namespace dicp {

class ReshapeOperation : public atb::Operation {
public:
    explicit ReshapeOperation(const std::string& name);
    ~ReshapeOperation(){};
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    std::string opName_;
};

}  // namespace dicp
