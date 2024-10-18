#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* LinearOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::LinearParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    DICP_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB << ", hasBias:" << param.hasBias
                   << ", outDataType:" << param.outDataType;
    atb::Operation* op = nullptr;
    ;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
