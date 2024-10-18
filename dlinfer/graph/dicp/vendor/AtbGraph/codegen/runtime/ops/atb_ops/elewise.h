#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* ElewiseOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::ElewiseParam param;
    if (paramJson.contains("elewiseType")) {
        auto tmp = paramJson["elewiseType"].get<int32_t>();
        param.elewiseType = static_cast<atb::infer::ElewiseParam::ElewiseType>(tmp);
    }
    if (paramJson.contains("quantParam")) {
        auto quantJson = paramJson["quantParam"];
        atb::infer::ElewiseParam::QuantParam quantParam;
        if (quantJson.contains("inputScale")) {
            quantParam.inputScale = quantJson["inputScale"].get<float>();
        }
        if (quantJson.contains("inputOffset")) {
            quantParam.inputOffset = quantJson["inputOffset"].get<int32_t>();
        }
        param.quantParam = quantParam;
    }
    if (paramJson.contains("mulsParam")) {
        auto mulsJson = paramJson["mulsParam"];
        atb::infer::ElewiseParam::MulsParam mulsParam;
        if (mulsJson.contains("varAttr")) {
            mulsParam.varAttr = mulsJson["varAttr"].get<float>();
        }
        param.mulsParam = mulsParam;
    }
    if (paramJson.contains("outTensorType")) {
        auto tmp = paramJson["outTensorType"].get<int32_t>();
        param.outTensorType = static_cast<aclDataType>(tmp);
    }
    DICP_LOG(INFO) << "ElewiseParam: elewiseType:" << param.elewiseType << ", outTensorType:" << param.outTensorType;
    atb::Operation* op = nullptr;
    ;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
