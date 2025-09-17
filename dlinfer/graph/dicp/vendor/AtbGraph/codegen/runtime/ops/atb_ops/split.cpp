#include "atb_ops.h"
#include "utils/common.h"

namespace dicp {

atb::Operation* SplitOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::SplitParam param;
    if (paramJson.contains("splitDim")) {
        param.splitDim = paramJson["splitDim"].get<int32_t>();
    }
    if (paramJson.contains("splitNum")) {
        param.splitNum = paramJson["splitNum"].get<int32_t>();
    }
    if (paramJson.contains("splitSizes")) {
        auto splitSizes = paramJson["splitSizes"].get<std::vector<int32_t>>();
        if (splitSizes.size() > 0) {
            param.splitSizes.resize(splitSizes.size());
            for (size_t i = 0; i < splitSizes.size(); ++i) {
                param.splitSizes[i] = splitSizes[i];
            }
        }
    }
    DICP_LOG(INFO) << "SplitParam: splitDim: " << param.splitDim << " splitNum: " << param.splitNum
                   << " splitSizes:" << svectorToString<int32_t>(param.splitSizes);
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("SplitOperation", SplitOperationCreate);

}  // namespace dicp
