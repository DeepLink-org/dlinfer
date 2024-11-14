#pragma once

#include <acl/acl.h>
#include <atb/types.h>

#include <sstream>
#include <string>

#include "atb/svector.h"
#include "utils/log.h"

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

template <typename T>
std::string svectorToString(const atb::SVector<T>& vec, const std::string& delimiter = ", ", const std::string& prefix = "[", const std::string& suffix = "]") {
    if (vec.empty()) {
        return prefix + suffix;
    }

    std::ostringstream oss;
    oss << prefix;

    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            oss << delimiter;
        }
        oss << vec[i];
    }

    oss << suffix;
    return oss.str();
}

}  // namespace dicp
