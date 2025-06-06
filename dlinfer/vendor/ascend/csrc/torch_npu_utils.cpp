// Copyright (c) 2024, DeepLink. All rights reserved.
#include <ATen/ScalarOps.h>
#include <c10/core/TensorImpl.h>

#include <chrono>
#include <ostream>
// using torch_npu acl headers in stead of cann's
// pre include before hccl/hccl.h to prevent mismatch between two vesions of acl.h
// clang-format off
#include <third_party/acl/inc/acl/acl.h>
#include <hccl/hccl.h>
// clang-format on
#include <torch_npu/csrc/core/NPUTensorImpl.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUException.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpParamMaker.h>

#include "torch_npu_utils.hpp"

std::unordered_map<SubModule, std::string> submoduleMap = {
    {SubModule::PTA, "PTA"}, {SubModule::OPS, "OPS"}, {SubModule::DIST, "DIST"}, {SubModule::GRAPH, "GRAPH"}, {SubModule::PROF, "PROF"}};

std::unordered_map<ErrCode, std::string> errCodeMap = {{ErrCode::SUC, "success"},
                                                       {ErrCode::PARAM, "invalid parameter"},
                                                       {ErrCode::TYPE, "invalid type"},
                                                       {ErrCode::VALUE, "invalid value"},
                                                       {ErrCode::PTR, "invalid pointer"},
                                                       {ErrCode::INTERNAL, "internal error"},
                                                       {ErrCode::MEMORY, "memory error"},
                                                       {ErrCode::NOT_SUPPORT, "feature not supported"},
                                                       {ErrCode::NOT_FOUND, "resource not found"},
                                                       {ErrCode::UNAVAIL, "resource unavailable"},
                                                       {ErrCode::SYSCALL, "system call failed"},
                                                       {ErrCode::TIMEOUT, "timeout error"},
                                                       {ErrCode::PERMISSION, "permission error"},
                                                       {ErrCode::ACL, "call acl api failed"},
                                                       {ErrCode::HCCL, "call hccl api failed"},
                                                       {ErrCode::GE, "call ge api failed"}};

static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);

    auto milli_time = std::chrono::duration_cast<std::chrono::milliseconds>(micros).count() % 1000;
    auto micro_time = micros.count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d-%H:%M:%S");
    return oss.str();
}

std::string formatErrorCode(SubModule submodule, ErrCode errorCode) {
    std::ostringstream oss;
    int deviceIndex = -1;
    c10_npu::GetDevice(&deviceIndex);
    char* rankId_val = std::getenv("RANK");
    int64_t rank_id = (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
    oss << "\n[ERROR] " << getCurrentTimestamp() << " (PID:" << getpid() << ", Device:" << deviceIndex << ", RankID:" << rank_id << ") ";
    oss << "ERR" << std::setw(2) << std::setfill('0') << static_cast<int>(submodule);
    oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode);
    oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];
    return oss.str();
}

namespace torch_npu {
// NPUStorageImpl
void NPUStorageImpl::release_resources() { StorageImpl::release_resources(); }
NPUStorageImpl::NPUStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::DataPtr data_ptr, at::Allocator* allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, at::DataPtr(std::move(data_ptr)), allocator, resizable) {}
// NPUTensorImpl
NPUTensorImpl::NPUTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(std::move(storage), c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1}, data_type) {
    is_non_overlapping_and_dense_ = false;
}
void NPUTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl){};
c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(const c10::VariableVersion& version_counter,
                                                                           bool allow_tensor_metadata_change) const {};
c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(c10::VariableVersion&& version_counter, bool allow_tensor_metadata_change) const {};
NPUTensorImpl::~NPUTensorImpl(){};
}  // namespace torch_npu

namespace dlinfer {
namespace ascend {

aclDataType convert_to_acl_data_type(const at::ScalarType& data_type) {
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported", OPS_ERROR(ErrCode::NOT_SUPPORT))
    return acl_dtype;
}

bool is_scalar_wrapped_to_tensor(const at::Tensor& tensor) {
    if (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor)) {
        return true;
    }
    return false;
}

at::Tensor copy_scalar_to_device(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type) {
    at::Tensor cpu_tensor = scalar_to_tensor(cpu_scalar).to(scalar_data_type);
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    return cpuPinMemTensor.to(c10::Device(c10::DeviceType::PrivateUse1, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

at::Tensor unsafe_empty_workspace(uint64_t workspace_size) {
    ASCEND_LOGD("Alloc workspace %zu bytes unsafely.", workspace_size);
    c10::Allocator* allocator = c10_npu::NPUCachingAllocator::get();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_npu::NPUStorageImpl>(
        c10::StorageImpl::use_byte_size_t(), workspace_size, allocator->allocate(workspace_size), allocator, true);
    static auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    return tensor;
}

#define GET_FUNC(funcName) funcName

aclError AclSetCompileopt(aclCompileOpt opt, const char* value) {
    typedef aclError (*aclSetCompileoptFunc)(aclCompileOpt opt, const char* value);
    static aclSetCompileoptFunc func = nullptr;
    if (func == nullptr) {
        func = (aclSetCompileoptFunc)GET_FUNC(aclSetCompileopt);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclSetCompileopt");
    auto ret = func(opt, value);
    return ret;
}

aclError AclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value) {
    typedef aclError (*AclrtCtxSetSysParamOptFunc)(aclSysParamOpt opt, int64_t value);
    static AclrtCtxSetSysParamOptFunc func = nullptr;
    if (func == nullptr) {
        // func = (AclrtCtxSetSysParamOptFunc)GET_FUNC(aclrtCtxSetSysParamOpt);
    }
    if (func == nullptr) {
        TORCH_WARN("Failed to find this aclrtCtxSetSysParamOpt function!");
        return ACL_ERROR_NONE;
    }
    auto ret = func(opt, value);
    return ret;
}

#define HCCL_CHECK_ERROR(cmd)                                                                                                                         \
    do {                                                                                                                                              \
        HcclResult error = cmd;                                                                                                                       \
        if (error != HCCL_SUCCESS) {                                                                                                                  \
            std::string err = "[ERROR] HCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + DIST_ERROR(ErrCode::HCCL) + ".\n"; \
            throw std::runtime_error(err);                                                                                                            \
        }                                                                                                                                             \
    } while (0)

static bool deterministicaclnn_oldstatus = false;
void SetDeterministic() {
    auto deterministicAlgorithmsStatus = at::globalContext().deterministicAlgorithms();
    if (deterministicaclnn_oldstatus != deterministicAlgorithmsStatus) {
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_DETERMINISTIC, deterministicAlgorithmsStatus ? "1" : "0"));
        NPU_CHECK_ERROR(AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0));
        HcclConfigValue configValue = {deterministicAlgorithmsStatus ? 1 : 0};
        HCCL_CHECK_ERROR(HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, configValue));
        deterministicaclnn_oldstatus = deterministicAlgorithmsStatus;
    }
}

}  // namespace ascend
}  // namespace dlinfer
