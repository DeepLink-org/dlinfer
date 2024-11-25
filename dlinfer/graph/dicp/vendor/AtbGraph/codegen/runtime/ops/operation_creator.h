#pragma once

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

#include "atb/operation.h"

namespace dicp {

using OperationCreateFunc = std::function<atb::Operation*(const nlohmann::json& paramJson)>;

std::unordered_map<std::string, OperationCreateFunc>& getGlobalFuncMap();

struct RegisterOp {
    RegisterOp(const std::string& name, OperationCreateFunc func) { getGlobalFuncMap()[name] = func; }
};

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE_NAME(prefix) CONCATENATE(prefix, __COUNTER__)

#define REGISTER_OPERATION(OpName, CreateFunc) static RegisterOp reg##OpName(#OpName, CreateFunc);
#define REGISTER_ATB_OPERATION(OpNameStr, CreateFunc) static RegisterOp MAKE_UNIQUE_NAME(reg_)(OpNameStr, CreateFunc);

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson);

}  // namespace dicp
