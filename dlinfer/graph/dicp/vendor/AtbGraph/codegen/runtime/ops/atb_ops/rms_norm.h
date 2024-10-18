#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* RmsNormOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::RmsNormParam param;
    param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    if (paramJson.contains("normParam")) {
        auto normParamJson = paramJson["normParam"];
        if (normParamJson.contains("epsilon")) {
            auto eps = normParamJson["epsilon"].get<float>();
            param.normParam.epsilon = eps;
        }
        if (normParamJson.contains("rstd")) {
            auto rstd = normParamJson["rstd"].get<bool>();
            param.normParam.rstd = rstd;
        }
    }
    DICP_LOG(INFO) << "RmsNormParam: layerType:" << param.layerType << ", epsilon:" << param.normParam.epsilon << ", rstd: " << param.normParam.rstd;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
