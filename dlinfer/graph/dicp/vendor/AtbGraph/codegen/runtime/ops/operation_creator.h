#pragma once

#include <nlohmann/json.hpp>

#include <iostream>
#include <unordered_map>
#include <string>

#include "atb/operation.h"

namespace dicp {

using OperationCreateFunc = std::function<atb::Operation*(const nlohmann::json& paramJson)>;

std::unordered_map<std::string, OperationCreateFunc>& getGlobalFuncMap();

struct RegisterOp {
    RegisterOp(const std::string& name, OperationCreateFunc func) {
        std::cout << "################# in RegisterOp: name: " << name << std::endl;
        getGlobalFuncMap()[name] = func;
    }
};

#define REGISTER_OPERATION(OpName, CreateFunc) \
    static RegisterOp reg##OpName(#OpName, CreateFunc);

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson);

}  // namespace dicp
