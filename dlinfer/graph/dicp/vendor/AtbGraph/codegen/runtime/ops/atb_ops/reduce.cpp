#pragma once
#include "atb_ops.h"
#include "utils/common.h"

namespace dicp {

atb::Operation* ReduceOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::ReduceParam param;
    if (paramJson.contains("reduceType")) {
        auto type = paramJson["reduceType"].get<int32_t>();
        param.reduceType = static_cast<atb::infer::ReduceParam::ReduceType>(type);
    }
    if (paramJson.contains("axis")) {
        auto axis = paramJson["axis"].get<std::vector<int64_t>>();
        param.axis.resize(axis.size());
        for (size_t i = 0; i < axis.size(); ++i) {
            param.axis[i] = axis[i];
        }
    }
    DICP_LOG(INFO) << "ReduceParam: reduceType: " << param.reduceType << ", axis:" << svectorToString<int64_t>(param.axis);
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("ReduceOperation", ReduceOperationCreate);

}  // namespace dicp
