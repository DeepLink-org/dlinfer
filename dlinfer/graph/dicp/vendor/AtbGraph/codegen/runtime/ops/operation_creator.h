#pragma once

#include <nlohmann/json.hpp>

#include <string>

#include "all_ops.h"
#include "atb/operation.h"

namespace dicp {

using OperationCreateFunc = std::function<atb::Operation*(const nlohmann::json& paramJson)>;

static std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"LinearOperation", &LinearOperationCreate},
    {"ElewiseOperation", &ElewiseOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"ReshapeAndCacheOperation", &ReshapeAndCacheOperationCreate},
    {"PagedAttentionOperation", &PagedAttentionOperationCreate},
    {"AddRmsNormOperation", &AclNnAddRmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"SplitOperation", &SplitOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"AclNnCatOperation", &AclNnCatOperationCreate},
    {"AclNnBatchMatMulOperation", &AclNnBatchMatMulOperationCreate},
    {"GatherOperation", &GatherOperationCreate},
    {"AclNnPermuteOperation", &AclNnPermuteOperationCreate},
    {"AclNnCastOperation", &AclNnCastOperationCreate},
};

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson);

}  // namespace dicp
