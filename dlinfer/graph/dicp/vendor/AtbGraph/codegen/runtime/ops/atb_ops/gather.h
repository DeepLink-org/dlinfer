#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* GatherOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::GatherParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    DICP_LOG(INFO) << "GatherParam: axis: " << param.axis;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
