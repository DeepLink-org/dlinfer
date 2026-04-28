#pragma once

#include <atb/types.h>

namespace dicp {

class Config {
public:
    Config();
    ~Config(){};
    uint64_t WorkspaceBufferSize();

private:
    uint64_t workspaceBufferSize_;
};

Config& GetConfig();

}  // namespace dicp
