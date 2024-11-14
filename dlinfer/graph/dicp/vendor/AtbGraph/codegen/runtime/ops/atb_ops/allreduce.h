#pragma once
#include "atb_ops.h"
namespace dicp {

inline atb::Operation* AllReduceOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::AllReduceParam param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int32_t>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int32_t>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int32_t>();
    }
    if (paramJson.contains("allReduceType")) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        auto tmp = paramJson["commMode"].get<int32_t>();
        param.commMode = static_cast<atb::infer::CommMode>(tmp);
    }
    if (paramJson.contains("commDomain")) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    DICP_LOG(INFO) << "AllReduceParam: rank:" << param.rank << ", rankSize:" << param.rankSize << ", backend:" << param.backend << ", allReduceType"
                   << param.allReduceType << ". commDomain" << param.commDomain;
    atb::Operation* op = nullptr;

    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

}  // namespace dicp
