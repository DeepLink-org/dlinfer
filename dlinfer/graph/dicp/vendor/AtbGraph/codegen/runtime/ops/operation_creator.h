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
    {"AclNnDivOperation", &AclNnDivOperationCreate},
    {"AclNnDivsOperation", &AclNnDivsOperationCreate},
    {"AclNnAddOperation", &AclNnAddOperationCreate},
    {"AclNnAddsOperation", &AclNnAddsOperationCreate},
    {"AclNnMulOperation", &AclNnMulOperationCreate},
    {"AclNnMulsOperation", &AclNnMulsOperationCreate},
    {"AclNnSubOperation", &AclNnSubOperationCreate},
    {"AclNnSubsOperation", &AclNnSubsOperationCreate},
    {"AclNnPowTensorScalarOperation", &AclNnPowTensorScalarOperationCreate},
    {"AclNnPowTensorTensorOperation", &AclNnPowTensorTensorOperationCreate},
    {"AclNnMaxOperation", &AclNnMaxOperationCreate},
    {"AclNnReciprocalOperation", &AclNnReciprocalOperationCreate},
    {"AclNnGatherOperation", &AclNnGatherOperationCreate},
    {"AclNnGeScalarOperation", &AclNnGeScalarOperationCreate},
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
    {"AclNnInplaceDivOperation", &AclNnInplaceDivOperationCreate},
    {"AclNnInplaceScatterOperation", &AclNnInplaceScatterOperationCreate},
    {"AclNnExpandOperation", &AclNnExpandOperationCreate},
    {"CustomViewOperation", &CustomViewOperationCreate},
    {"CustomUnsqueezeOperation", &CustomUnsqueezeOperationCreate},
    {"CustomSqueezeOperation", &CustomSqueezeOperationCreate},
    {"CustomScalarTensorOperation", &CustomScalarTensorOperationCreate},
};

atb::Operation* CreateOperation(const std::string& opName, const nlohmann::json& paramJson);

}  // namespace dicp
