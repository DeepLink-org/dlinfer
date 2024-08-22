// copy from op-plugin/op_plugin/utils/op_api_common.h
// op-plugin commit: 1d92099b46704088c9cb18e4675057f361137b79
#pragma once

#include <type_traits>
#include <tuple>
#include <dlfcn.h>

#include <ATen/Tensor.h>
#include <third_party/acl/inc/acl/acl_base.h>
// #include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "torch_npu_utils.hpp"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

typedef aclTensor *(*_aclCreateTensor)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t *stride, int64_t offset, aclFormat format,
                                       const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value, uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value, uint64_t size);
typedef aclScalarList *(*_aclCreateScalarList)(const aclScalar *const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);
typedef int (*_aclDestroyScalarList)(const aclScalarList *array);

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

#define MEMCPY_TO_BUF(data_expression, size_expression)                                                                \
    if (g_hash_offset + (size_expression) > g_hash_buf_size) {                                                         \
        g_hash_offset = g_hash_buf_max_size;                                                                           \
        return;                                                                                                        \
    }                                                                                                                  \
    memcpy(g_hash_buf + g_hash_offset, data_expression, size_expression);                                              \
    g_hash_offset += size_expression;

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

#define GET_OP_API_FUNC_FROM_FEATURE_LIB(lib_handler, lib_name)                                                        \
    static auto lib_handler = GetOpApiLibHandler(lib_name);                                                            \
    if ((lib_handler) != nullptr) {                                                                                      \
        auto funcAddr = GetOpApiFuncAddrInLib(lib_handler, lib_name, api_name);                                        \
        if (funcAddr != nullptr) {                                                                                     \
            return funcAddr;                                                                                           \
        }                                                                                                              \
    }

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name);

inline void *GetOpApiFuncAddr(const char *apiName)
{
    static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return GetOpApiFuncAddrFromFeatureLib(apiName);
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported", OPS_ERROR(ErrCode::PARAM));
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = dlinfer::ascend::convert_to_acl_data_type(scalar_data_type);
    c10::SmallVector<int64_t, 5> storageDims;
    // if acl_data_type is ACL_STRING, storageDims is empty.
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
            OPS_ERROR(ErrCode::VALUE));
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }

    if (dlinfer::ascend::is_scalar_wrapped_to_tensor(at_tensor)) {
        c10::Scalar expScalar = at_tensor.item();
        at::Tensor aclInput = dlinfer::ascend::copy_scalar_to_device(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = dlinfer::ascend::convert_to_acl_data_type(scalar_data_type);
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

template <std::size_t N> inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclScalarList *ConvertType(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertType(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }

    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return dlinfer::ascend::convert_to_acl_data_type(scalarType);
}

template <typename T> T ConvertType(T value)
{
    return value;
}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

inline void Release(aclScalarList *p)
{
    static const auto aclDestroyScalarList = GET_OP_API_FUNC(aclDestroyScalarList);
    if (aclDestroyScalarList == nullptr) {
        return;
    }

    aclDestroyScalarList(p);
}

template <typename T> void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I> void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple> void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I> auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
    typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple> auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

template <std::size_t N> void add_param_to_buf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), static_cast<int64_t>(value.size() * sizeof(bool)));
}

template <typename T> void add_param_to_buf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void add_param_to_buf(const at::Tensor &);
void add_param_to_buf(const at::Scalar &);
void add_param_to_buf(const at::IntArrayRef &);
void add_param_to_buf(const at::ArrayRef<bool> &);
void add_param_to_buf(const at::TensorList &);
void add_param_to_buf(const at::ArrayRef<at::Scalar> &);
void add_param_to_buf(const c10::optional<at::Tensor> &);
void add_param_to_buf(const c10::optional<at::IntArrayRef> &);
void add_param_to_buf(const c10::optional<at::Scalar> &);
void add_param_to_buf(const at::ScalarType);
void add_param_to_buf(const string &);
void add_param_to_buf();

template <typename T, typename... Args> void add_param_to_buf(const T &arg, Args &...args)
{
    add_param_to_buf(arg);
    add_param_to_buf(args...);
}

uint64_t calc_hash_id();

#define DO_COMPATIBILITY(aclnn_api, originCallExpression)                                                              \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr) {                                         \
            ASCEND_LOGW("%s or %sGetWorkspaceSize not in %s, or %s not found. Will call %s", #aclnn_api, #aclnn_api,   \
                        GetOpApiLibName(), GetOpApiLibName(), #originCallExpression);                                  \
            return originCallExpression;                                                                               \
        }                                                                                                              \
    } while (0)

typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);
typedef aclOpExecutor *(*PTAGetExecCache)(uint64_t, uint64_t *);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef bool (*CanUsePTACache)(const char *);
typedef void (*UnInitPTACacheThreadLocal)();

inline void UnInitCacheThreadLocal()
{
    static const auto unInitPTACacheThreadLocalAddr = GetOpApiFuncAddr("UnInitPTACacheThreadLocal");
    UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
        reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
    if (unInitPTACacheThreadLocalFunc) {
        unInitPTACacheThreadLocalFunc();
    }
}

template <typename... Args> bool hit_cache(aclrtStream acl_stream, const char *aclnn_api, void *phrase2, Args &&...args)
{
    static const auto ptaGetExecCacheAddr = GetOpApiFuncAddr("PTAGetExecCache");
    static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
    static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");
    static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");
    PTAGetExecCache ptaGetExecCacheFunc = reinterpret_cast<PTAGetExecCache>(ptaGetExecCacheAddr);
    InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
        reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
    SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);
    CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);
    bool has_func = ptaGetExecCacheFunc && initPTACacheThreadLocalFunc && setPTAHashKeyFunc;
    bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(aclnn_api);
    if (!has_func || !can_use) {
        return false;
    }
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    initPTACacheThreadLocalFunc();
    g_hash_offset = 0;
    add_param_to_buf(std::string(aclnn_api), args...);
    uint64_t hashId = calc_hash_id();
    setPTAHashKeyFunc(hashId);
    aclOpExecutor *executor = ptaGetExecCacheFunc(hashId, workspace_size_addr);
    if (executor == nullptr) {
        return false;
    }
    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        auto workspace_tensor = dlinfer::ascend::unsafe_empty_workspace(workspace_size);
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }
    auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, phrase2]() -> int {
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(api_ret == 0, "call failed, detail:", aclGetRecentErrMsg(), OPS_ERROR(ErrCode::INTERNAL));
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(aclnn_api);
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    UnInitCacheThreadLocal();
    return true;
}

/**
 * check arg is at::Tensor ?
 */
template<typename T>
struct is_at_tensor : std::false_type {};

template<>
struct is_at_tensor<at::Tensor> : std::true_type {};

/**
 * check arg is at::TensorList ?
 */
template<typename T>
struct is_at_tensor_list : std::false_type {};

template<>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

/**
 * find first at::Tensor
 */
template <std::size_t I = 0, typename...Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...>& t, at::Tensor& res) {}

template <std::size_t I = 0, typename... Ts>
typename std::enable_if < I<sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

/**
 * get the device
 */
template <typename... Ts>
auto DecodeDevice(Ts&... args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

/**
 * 异步调用npu执行, 无返回值.
 */
#define EXEC_NPU_CMD(aclnn_api, ...)                                                                                   \
    do {                                                                                                               \
        auto device = DecodeDevice(__VA_ARGS__);                                                                       \
        /* c10_npu::NPUGuard guard(device); */                                                                              \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        if (hit_cache(acl_stream, #aclnn_api, opApiFuncAddr, __VA_ARGS__)) {                                           \
            break;                                                                                                     \
        }                                                                                                              \
        dlinfer::ascend::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                \
            OPS_ERROR(ErrCode::INTERNAL));                                                                             \
        void *workspace_addr = nullptr;                                                                                \
        if (workspace_size != 0) {                                                                                     \
            auto workspace_tensor = dlinfer::ascend::unsafe_empty_workspace(workspace_size);             \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {            \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                     \
                OPS_ERROR(ErrCode::INTERNAL));                                                                         \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand cmd;                                                                                 \
        cmd.Name(#aclnn_api);                                                                                          \
        cmd.SetCustomHandler(acl_call);                                                                                \
        cmd.Run();                                                                                                     \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

/**
 * 针对aclnnInplaceCopy重载EXEC_NPU_CMD, 获取第二个tensor(src)的device
 */
#define EXEC_NPU_COPY_CMD(aclnn_api, ...)                                                                              \
    do {                                                                                                               \
        auto args_tuple = std::make_tuple(__VA_ARGS__);                                                                \
        auto second_tensor = std::get<1>(args_tuple);                                                                  \
        auto device = second_tensor.device();                                                                          \
        /* c10_npu::NPUGuard guard(device); */                                                                             \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.");                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        if (hit_cache(acl_stream, #aclnn_api, opApiFuncAddr, __VA_ARGS__)) {                                           \
            break;                                                                                                     \
        }                                                                                                              \
        dlinfer::ascend::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());               \
        void *workspace_addr = nullptr;                                                                                \
        if (workspace_size != 0) {                                                                                     \
            auto workspace_tensor = dlinfer::ascend::unsafe_empty_workspace(workspace_size);             \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {              \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                    \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand cmd;                                                                                 \
        cmd.Name(#aclnn_api);                                                                                          \
        cmd.SetCustomHandler(acl_call);                                                                                \
        cmd.Run();                                                                                                     \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

#define EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnn_api, ...)                                                                   \
    do {                                                                                                               \
        auto device = DecodeDevice(__VA_ARGS__);                                                                       \
        /* c10_npu::NPUGuard guard(device); */                                                                              \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        dlinfer::ascend::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                \
            OPS_ERROR(ErrCode::INTERNAL));                                                                             \
        void *workspace_addr = nullptr;                                                                                \
        if (workspace_size != 0) {                                                                                     \
            auto workspace_tensor = dlinfer::ascend::unsafe_empty_workspace(workspace_size);             \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {            \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                     \
                OPS_ERROR(ErrCode::INTERNAL));                                                                         \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand cmd;                                                                                 \
        cmd.Name(#aclnn_api);                                                                                          \
        cmd.SetCustomHandler(acl_call);                                                                                \
        cmd.Run();                                                                                                     \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

template <typename Tuple> class ConvertedParams {
public:
    explicit ConvertedParams(Tuple &&convertedParams, ReleaseHugeMem releaseMemFunc,
                             UnInitHugeMemThreadLocal unInitMemFunc) : convertedParams_(std::move(convertedParams)),
                                                                       releaseMemFunc_(releaseMemFunc),
                                                                       unInitMemFunc_(unInitMemFunc){};
    ConvertedParams(ConvertedParams &&other) : convertedParams_(std::move(other.convertedParams_))
    {
        other.validParams_ = false;
    };
    ConvertedParams &operator=(ConvertedParams &&other)
    {
        if (this == &other) {
            return *this;
        }

        convertedParams_ = std::move(other.convertedParams_);
        validParams_ = true;
        other.validParams_ = false;
        return *this;
    }

    ConvertedParams() = delete;
    ConvertedParams(const ConvertedParams &other) = delete;
    ConvertedParams &operator=(const ConvertedParams &other) = delete;

    ~ConvertedParams()
    {
        if (validParams_) {
            ReleaseConvertTypes(convertedParams_);
            if (releaseMemFunc_) {
                releaseMemFunc_(nullptr, false);
            }
            if (unInitMemFunc_) {
                unInitMemFunc_(nullptr, false);
            }
        }
    }

    const Tuple &GetConvertedParams() const
    {
        return convertedParams_;
    }

    template <size_t i> auto Get()
    {
        return std::get<i>(convertedParams_);
    }

private:
    Tuple convertedParams_;
    ReleaseHugeMem releaseMemFunc_ = nullptr;
    UnInitHugeMemThreadLocal unInitMemFunc_ = nullptr;
    bool validParams_{true};
};

/**
 * 同步调用npu执行，返回把aten的tensor, scalar, array等转换后的参数,
 */
#define EXEC_NPU_CMD_SYNC(aclnn_api, ...)                                                                              \
    [](const char *apiName, const char *workspaceSizeApiName, auto &...args) -> auto {                                 \
        auto device = DecodeDevice(args...);                                                                           \
        /* c10_npu::NPUGuard guard(device); */                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiName);                           \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(apiName);                                                   \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " and ",              \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                              \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        dlinfer::ascend::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);                             \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                \
            OPS_ERROR(ErrCode::INTERNAL));                                                                             \
        void *workspace_addr = nullptr;                                                                                \
        if (workspace_size != 0) {                                                                                     \
            auto workspace_tensor = dlinfer::ascend::unsafe_empty_workspace(workspace_size);             \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor, apiName]() -> int {   \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg(),                     \
                OPS_ERROR(ErrCode::INTERNAL));                                                                         \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand cmd;                                                                                 \
        cmd.Name(apiName);                                                                                             \
        cmd.SetCustomHandler(acl_call);                                                                                \
        cmd.Run();                                                                                                     \
        UnInitCacheThreadLocal();                                                                                      \
        cmd.Sync();                                                                                                    \
        return ConvertedParams<decltype(converted_params)>(std::move(converted_params),                                \
            releaseMemFunc, unInitMemFunc);                                                                            \
    }(#aclnn_api, #aclnn_api "GetWorkspaceSize", __VA_ARGS__)
