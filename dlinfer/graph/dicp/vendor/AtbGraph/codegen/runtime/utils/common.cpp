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

aclDataType get_acl_dtype(const std::string& dtype) {
    if (dtype == "INT64") {
        return ACL_INT64;
    } else if (dtype == "INT32") {
        return ACL_INT32;
    } else if (dtype == "FLOAT") {
        return ACL_FLOAT;
    } else if (dtype == "FLOAT16") {
        return ACL_FLOAT16;
    } else if (dtype == "BF16") {
        return ACL_BF16;
    } else {
        throw std::invalid_argument("Unsupported dtype: " + dtype);
    }
}

}  // namespace dicp
