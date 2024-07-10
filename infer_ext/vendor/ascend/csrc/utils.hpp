#include <dlfcn.h>
#include <acl/acl.h>
#include <torch_npu/csrc/npu/Stream.h>

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        printf("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        printf("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

#define EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnn_api, ...)                                                                   \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
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
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());               \
        void *workspace_addr = nullptr;                                                                                \
        if (workspace_size != 0) {                                                                                     \
            auto workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);             \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {            \
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
    } while (false)