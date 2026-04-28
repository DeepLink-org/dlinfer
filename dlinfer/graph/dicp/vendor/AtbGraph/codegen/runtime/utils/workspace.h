#pragma once

#include <atb/types.h>
#include <torch/torch.h>

#include <vector>

namespace dicp {

class Workspace {
public:
    Workspace();
    ~Workspace(){};
    void* GetWorkspaceBuffer(uint64_t bufferSize);

private:
    torch::Tensor CreateAtTensor(uint64_t bufferSize);

private:
    void* buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
    torch::Tensor atTensor_;
};

void* GetWorkspaceBuffer(uint64_t bufferSize);

}  // namespace dicp
