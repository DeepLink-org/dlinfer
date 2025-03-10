#include "atb_ops.h"

namespace dicp {

inline atb::Operation* ReshapeAndCacheOperationCreate([[maybe_unused]] const nlohmann::json& paramJson) {
    atb::infer::ReshapeAndCacheParam param;
    if (paramJson.contains("KvCacheCfg")) {
        auto value = paramJson["KvCacheCfg"].get<int32_t>();
        param.kvCacheCfg = static_cast<atb::infer::ReshapeAndCacheParam::KvCacheCfg>(value);
    }
    DICP_LOG(INFO) << "ReshapeAndCacheParam: {}";
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("ReshapeAndCacheOperation", ReshapeAndCacheOperationCreate);

}  // namespace dicp
