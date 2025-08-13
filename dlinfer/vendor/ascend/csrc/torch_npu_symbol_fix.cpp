#include <torch_npu/csrc/core/npu/interface/AclInterface.h>
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>

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

void record_mem_hbm_ecc_error() {
    throw std::runtime_error(
        "Dlinfer record_mem_hbm_ecc_error should not be called. "
        "Please check your environment setup.");
}

#if defined(TORCH_VERSION_2_6_0)
std::string handleDeviceError(int errorCode) {
    throw std::runtime_error(
        "Dlinfer handleDeviceError should not be called. "
        "Please check your environment setup.");
    return "";
}
namespace option {
bool OptionsManager::IsCompactErrorOutput() {
    throw std::runtime_error(
        "Dlinfer IsCompactErrorOutput should not be called. "
        "Please check your environment setup.");
}
}  // namespace option
#endif

#if !defined(TORCH_VERSION_2_7_1)
bool checkUceErrAndRepair(bool check_error, std::string& err_msg) {
    throw std::runtime_error(
        "Dlinfer checkUceErrAndRepair should not be called. "
        "Please check your environment setup.");
    return false;
}
#endif

#if defined(TORCH_VERSION_2_7_1)
namespace option {
bool OptionsManager::ShouldPrintLessError() {
    throw std::runtime_error(
        "Dlinfer record_mem_hbm_ecc_error should not be called. "
        "Please check your environment setup.");
}
}  // namespace option

std::string handleDeviceError(int errorCode) {
    throw std::runtime_error(
        "Dlinfer record_mem_hbm_ecc_error should not be called. "
        "Please check your environment setup.");
    return "";
}
#endif

}  // namespace c10_npu
