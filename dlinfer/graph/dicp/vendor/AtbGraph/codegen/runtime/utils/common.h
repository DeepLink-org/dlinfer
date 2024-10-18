#pragma once

#include <acl/acl.h>
#include <atb/types.h>

#include "log.h"

namespace dicp {

#define DICP_CHECK_RET(call)                                                                                                      \
    do {                                                                                                                          \
        int ret = (call);                                                                                                         \
        if (ret != 0) {                                                                                                           \
            DICP_LOG(ERROR) << "Error: " << #call << " failed with return code " << ret << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error("check call failed");                                                                        \
        }                                                                                                                         \
    } while (0)

#define DICP_CHECK_ATB_RET(call)                                                                                                  \
    do {                                                                                                                          \
        int ret = (call);                                                                                                         \
        if (ret != atb::NO_ERROR) {                                                                                               \
            DICP_LOG(ERROR) << "Error: " << #call << " failed with return code " << ret << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error("ATB call failed");                                                                          \
        }                                                                                                                         \
    } while (0)

}  // namespace dicp
