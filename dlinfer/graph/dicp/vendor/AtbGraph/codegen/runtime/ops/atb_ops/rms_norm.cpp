#include "atb_ops.h"

namespace dicp {

atb::Operation* RmsNormOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::RmsNormParam param;
    param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    float eps = 0.0;
    if (paramJson.contains("layerType")) {
        auto layerType = paramJson["layerType"].get<int>();
        param.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(layerType);
    }
    if (param.layerType == atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM && paramJson.contains("normParam")) {
        auto normParamJson = paramJson["normParam"];
        if (normParamJson.contains("epsilon")) {
            eps = normParamJson["epsilon"].get<float>();
            param.normParam.epsilon = eps;
        }
        if (normParamJson.contains("rstd")) {
            auto rstd = normParamJson["rstd"].get<bool>();
            param.normParam.rstd = rstd;
        }
    }
    if (param.layerType == atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM && paramJson.contains("preNormParam")) {
        auto preNormParamJson = paramJson["preNormParam"];
        if (preNormParamJson.contains("epsilon")) {
            eps = preNormParamJson["epsilon"].get<float>();
            param.preNormParam.epsilon = eps;
        }
    }
    DICP_LOG(INFO) << "RmsNormParam: layerType:" << param.layerType << ", epsilon:" << eps;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("RmsNormOperation", RmsNormOperationCreate);

}  // namespace dicp
