#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* SortOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SortParam param;
    if (paramJson.contains("num")) {
        auto tmp = paramJson["num"].get<int32_t>();
        param.num.resize(1);
        param.num[0] = tmp;
    }
    DICP_LOG(INFO) << "SortParam: topk:" << param.num[0];
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
