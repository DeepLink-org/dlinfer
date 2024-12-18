#include "atb_ops.h"

namespace dicp {

atb::Operation* RopeOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::RopeParam param;
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int32_t>();
    }
    if (paramJson.contains("cosFormat")) {
        param.cosFormat = paramJson["cosFormat"].get<int32_t>();
    }
    DICP_LOG(INFO) << "RopeParam: rotaryCoeff:" << param.rotaryCoeff << ", cosFormat:" << param.cosFormat;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("RopeOperation", RopeOperationCreate);

}  // namespace dicp
