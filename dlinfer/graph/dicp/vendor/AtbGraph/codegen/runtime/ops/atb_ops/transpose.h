#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* TransposeOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::TransposeParam param;
    if (paramJson.contains("perm")) {
        auto tmp = paramJson["perm"].get<std::vector<int32_t>>();
        param.perm.resize(tmp.size());
        for (unsigned int i = 0; i < tmp.size(); ++i) {
            param.perm[i] = tmp[i];
        }
    }
    DICP_LOG(INFO) << "TransposeParam: perm: " << param.perm;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
