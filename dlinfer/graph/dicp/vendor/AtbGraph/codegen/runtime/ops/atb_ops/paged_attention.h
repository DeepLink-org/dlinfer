#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* PagedAttentionOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::PagedAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int32_t>();
    }
    if (paramJson.contains("maskType")) {
        auto value = paramJson["maskType"].get<int32_t>();
        param.maskType = static_cast<atb::infer::PagedAttentionParam::MaskType>(value);
    }
    DICP_LOG(INFO) << "PagedAttentionParam: headNum" << param.headNum << " kvHeadNum: " << param.kvHeadNum << " qkScale: " << param.qkScale
                   << " maskType: " << param.maskType;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
