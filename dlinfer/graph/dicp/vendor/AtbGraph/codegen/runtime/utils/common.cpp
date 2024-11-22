#include "utils/common.h"

namespace dicp {

std::string atbDimsToString(const atb::Dims& d) {
    std::ostringstream oss;
    oss << "[";
    for (uint64_t i = 0; i < d.dimNum; ++i) {
        oss << d.dims[i];
        if (i < d.dimNum - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

}  // namespace dicp
