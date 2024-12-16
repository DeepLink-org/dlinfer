#include "atb_ops.h"

namespace dicp {

[[maybe_unused]] atb::Operation* ConcatOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::ConcatParam param;
    if (paramJson.contains("concatDim")) {
        param.concatDim = paramJson["concatDim"].get<int32_t>();
    }
    DICP_LOG(INFO) << "ConcatParam: concatDIm: " << param.concatDim;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("ConcatOperation", ConcatOperationCreate);

}  // namespace dicp
