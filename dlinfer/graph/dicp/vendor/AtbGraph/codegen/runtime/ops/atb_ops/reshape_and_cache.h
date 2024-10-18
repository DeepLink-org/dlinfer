#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* ReshapeAndCacheOperationCreate([[maybe_unused]] const nlohmann::json& paramJson) {
    atb::infer::ReshapeAndCacheParam param;
    DICP_LOG(INFO) << "ReshapeAndCacheParam: {}";
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
