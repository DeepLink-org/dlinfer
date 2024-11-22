#include "reshape_operation.h"

#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

ReshapeOperation::ReshapeOperation(const std::string& name) : opName_(name) {}

std::string ReshapeOperation::GetName() const { return opName_; }

atb::Status ReshapeOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) { return atb::NO_ERROR; }

atb::Status ReshapeOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
    return atb::NO_ERROR;
}

uint32_t ReshapeOperation::GetInputNum() const { return NUM1; }

uint32_t ReshapeOperation::GetOutputNum() const { return NUM1; }

}  // namespace dicp
