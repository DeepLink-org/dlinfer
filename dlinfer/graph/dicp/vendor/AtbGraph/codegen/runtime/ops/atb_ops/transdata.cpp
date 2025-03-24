#include "atb_ops.h"

namespace dicp {

atb::Operation* TransdataOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::TransdataParam param;
    if (paramJson.contains("transdataType")) {
        auto value =  paramJson["transdataType"].get<int32_t>();
        param.transdataType = static_cast<atb::infer::TransdataParam::TransdataType>(value);
    }
    DICP_LOG(INFO) << "TransdataParam: transdataType: " << param.transdataType;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("TransdataOperation", TransdataOperationCreate);

}  // namespace dicp
