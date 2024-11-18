#include "utils/config.h"

#include <cstdlib>
#include <string>

#include "utils/log.h"

namespace dicp {

constexpr int GB_1 = 1024 * 1024 * 1024;

Config::Config() {
    const char* envBufferSize = std::getenv("DICP_WORKSPACE_BUFFER_SIZE");
    if (envBufferSize) {
        workspaceBufferSize_ = std::stoull(envBufferSize);
    } else {
        workspaceBufferSize_ = 1 * GB_1;
    }
}

uint64_t Config::WorkspaceBufferSize() { return workspaceBufferSize_; }

Config& GetConfig() {
    static Config config;
    return config;
}

}  // namespace dicp
