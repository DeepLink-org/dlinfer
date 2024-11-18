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
    {"SplitWithSizeOperation", &AclNnSplitWithSizeOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"AclNnCatOperation", &AclNnCatOperationCreate},
    {"AclNnBatchMatMulOperation", &AclNnBatchMatMulOperationCreate},
    {"GatherOperation", &GatherOperationCreate},
    {"AclNnPermuteOperation", &AclNnPermuteOperationCreate},
    {"AclNnCastOperation", &AclNnCastOperationCreate},
    {"AclNnDivsOperation", &AclNnDivsOperationCreate},
    {"AclNnAddsOperation", &AclNnAddsOperationCreate},
    {"AclNnMulsOperation", &AclNnMulsOperationCreate},
    {"AclNnSubsOperation", &AclNnSubsOperationCreate},
    {"AclNnPowTensorScalarOperation", &AclNnPowTensorScalarOperationCreate},
    {"AclNnPowTensorTensorOperation", &AclNnPowTensorTensorOperationCreate},
    {"AclNnMaxOperation", &AclNnMaxOperationCreate},
    {"AclNnReciprocalOperation", &AclNnReciprocalOperationCreate},
    {"AclNnGtScalarOperation", &AclNnGtScalarOperationCreate},
    {"AclNnSWhereOperation", &AclNnSWhereOperationCreate},
    {"AclNnArangeOperation", &AclNnArangeOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"SoftmaxOperation", &SoftmaxOperationCreate},
    {"SortOperation", &SortOperationCreate},
    {"SliceOperation", &SliceOperationCreate},
    {"AclNnIndexSelectOperation", &AclNnIndexSelectOperationCreate},
    {"AclNnSliceOperation", &AclNnSliceOperationCreate},
    {"AclNnSoftmaxOperation", &AclNnSoftmaxOperationCreate},
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"AclNnTopkOperation", &AclNnTopkOperationCreate},
};

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson);

}  // namespace dicp
