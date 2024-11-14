#pragma once
#include "atb_ops.h"

namespace dicp {

inline atb::Operation* SoftmaxOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SoftmaxParam param;
    if (paramJson.contains("axes")) {
        auto tmp = paramJson["axes"].get<std::vector<int64_t>>();
        param.axes.resize(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
            param.axes[i] = tmp[i];
        }
    }
    DICP_LOG(INFO) << "SoftmaxParam: axes.size:" << param.axes.size() << " axes0: " << param.axes[0];
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
