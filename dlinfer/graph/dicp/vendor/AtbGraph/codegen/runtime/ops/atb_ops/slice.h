#pragma once
#include "atb_ops.h"
#include "utils/common.h"

namespace dicp {

inline atb::Operation* SliceOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SliceParam param;
    if (paramJson.contains("offsets")) {
        auto tmp = paramJson["offsets"].get<std::vector<int64_t>>();
        param.offsets.resize(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
            param.offsets[i] = tmp[i];
        }
    }
    if (paramJson.contains("size")) {
        auto tmp = paramJson["size"].get<std::vector<int64_t>>();
        param.size.resize(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
            param.size[i] = tmp[i];
        }
    }

    DICP_LOG(INFO) << "SliceParam: offsets:" << svectorToString<int64_t>(param.offsets) << ", size:" << svectorToString<int64_t>(param.size);
    atb::Operation* op = nullptr;

    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
