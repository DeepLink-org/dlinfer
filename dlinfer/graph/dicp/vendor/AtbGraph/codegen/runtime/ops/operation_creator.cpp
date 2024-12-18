#include "ops/operation_creator.h"

#include "utils/log.h"

namespace dicp {

std::unordered_map<std::string, OperationCreateFunc>& getGlobalFuncMap() {
    static std::unordered_map<std::string, OperationCreateFunc> funcMap;
    return funcMap;
}

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson) {
    auto g_funcMap = getGlobalFuncMap();
    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        DICP_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception& e) {
        DICP_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}

}  // namespace dicp
