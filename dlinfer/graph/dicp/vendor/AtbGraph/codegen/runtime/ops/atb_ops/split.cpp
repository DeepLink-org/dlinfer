#include "atb_ops.h"

namespace dicp {

atb::Operation* SplitOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SplitParam param;
    if (paramJson.contains("splitDim")) {
        param.splitDim = paramJson["splitDim"].get<int32_t>();
    }
    if (paramJson.contains("splitNum")) {
        param.splitNum = paramJson["splitNum"].get<int32_t>();
    }
    DICP_LOG(INFO) << "SplitParam: splitDim: " << param.splitDim << " splitNum: " << param.splitNum;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("SplitOperation", SplitOperationCreate);

}  // namespace dicp
