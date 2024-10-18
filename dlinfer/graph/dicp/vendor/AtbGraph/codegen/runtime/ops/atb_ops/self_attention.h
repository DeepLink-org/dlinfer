#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* SelfAttentionOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SelfAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int32_t>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("qScale")) {
        param.qScale = paramJson["qScale"].get<float>();
    }
    if (paramJson.contains("calcType")) {
        auto value = paramJson["calcType"].get<int32_t>();
        param.calcType = static_cast<atb::infer::SelfAttentionParam::CalcType>(value);
    }
    if (paramJson.contains("kernelType")) {
        auto value = paramJson["kernelType"].get<int32_t>();
        param.kernelType = static_cast<atb::infer::SelfAttentionParam::KernelType>(value);
    }
    if (paramJson.contains("clampType")) {
        auto value = paramJson["clampType"].get<int32_t>();
        param.clampType = static_cast<atb::infer::SelfAttentionParam::ClampType>(value);
    }
    if (paramJson.contains("isTriuMask")) {
        auto value = paramJson["isTriuMask"].get<int32_t>();
        param.isTriuMask = static_cast<uint32_t>(value);
    }
    if (paramJson.contains("maskType")) {
        auto value = paramJson["maskType"].get<int32_t>();
        param.maskType = static_cast<atb::infer::SelfAttentionParam::MaskType>(value);
    }
    DICP_LOG(INFO) << "SelfAttentionParam: headNum: " << param.headNum << " kvHeadNum: " << param.kvHeadNum << " calcType: " << param.calcType
                   << " kernelType: " << param.kernelType << " clampType: " << param.clampType << " qkScale: " << param.qkScale << " qScale: " << param.qScale
                   << " isTriuMask: " << param.isTriuMask << " maskType: " << param.maskType;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
