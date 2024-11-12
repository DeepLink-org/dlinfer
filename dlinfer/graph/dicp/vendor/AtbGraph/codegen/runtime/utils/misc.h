#pragma once
#include <sstream>
#include <string>

#include "atb/svector.h"
namespace dicp {
namespace utils {

void* GetCurrentStream();
int GetNewModelId();

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

}  // namespace utils
}  // namespace dicp
