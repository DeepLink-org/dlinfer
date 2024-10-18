#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* ActivationOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        auto value = paramJson["activationType"].get<int32_t>();
        param.activationType = static_cast<atb::infer::ActivationType>(value);
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    if (paramJson.contains("dim")) {
        param.dim = paramJson["dim"].get<int32_t>();
    }
    DICP_LOG(INFO) << "ActivationParam:  activationType: " << param.activationType << " scale:" << param.scale << " dim:" << param.dim;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
