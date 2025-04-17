#include <torch_npu/csrc/core/npu/interface/AclInterface.h>

#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "acl/acl_rt.h"

namespace c10_npu {
namespace acl {

// These functions are reimplemented to handle the missing symbol issue in
// torch-npu >= 2.3.1. If these functions are called, it indicates an environment
// setup issue and the program should terminate

aclError AclrtPeekAtLastError(aclrtLastErrLevel flag) {
    throw std::runtime_error(
        "Dlinfer AclrtPeekAtLastError should not be called. "
        "Please check your environment setup.");
    return ACL_ERROR;
}
}  // namespace acl

bool checkUceErrAndRepair() {
    throw std::runtime_error(
        "Dlinfer checkUceErrAndRepair should not be called. "
        "Please check your environment setup.");
    return false;
}

}  // namespace c10_npu
