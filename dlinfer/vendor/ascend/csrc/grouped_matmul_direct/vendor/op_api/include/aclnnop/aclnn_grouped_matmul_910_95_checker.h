/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_910_95_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_910_95_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_grouped_matmul_util.h"

namespace gmm {
template<typename T>
class AclnnDlinferGroupedMatmulDirect91095Checker {
public:
    explicit AclnnDlinferGroupedMatmulDirect91095Checker(const DlinferGroupedMatmulDirectParamsBase<T> &gmmParams) : gmmParams_(gmmParams){};
    ~AclnnDlinferGroupedMatmulDirect91095Checker(){};
    aclnnStatus CheckDlinferGroupedMatmulDirect91095() const;
    bool IsPerTileQuantMode() const;
    void SetInputName(const std::string& xName, const std::string& weightName, const std::string& perTokenScaleName,
                        const std::string& scaleName, const std::string& groupTensorName);

private:
    struct TensorDimInfo {
        size_t xDimNum = 0;
        size_t weightDimNum = 0;
        size_t scaleDimNum = 0;
        size_t pertokenScaleDimNum = 0;
        int64_t groupNum = 0;
        size_t biasDimNum = 0;
    };

    bool IsQuant(DataType &xDtype, DataType &weightDtype) const;
    aclnnStatus CheckGeneralQuantShape() const;
    aclnnStatus CheckQuantCasesFormat() const;

    aclnnStatus CheckDlinferGroupedMatmulDirectMxDtype() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectPerGroupDim() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectMxShape() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectMxScaleTranspose() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectPerTile() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectPerTileShape() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectMxfp8() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectMxfp4() const;
    aclnnStatus CheckDlinferGroupedMatmulDirectFp4MxDimValue() const;

    aclnnStatus CheckNonPerGroupQuantDim() const;
    aclnnStatus CheckNonPerGroupQuantPertokenShape() const;
    aclnnStatus CheckNonPerGroupQuantShape() const;
    aclnnStatus CheckInt8QuantDtype() const;
    aclnnStatus CheckInt8QuantParams() const;
    aclnnStatus CheckFp8Hif8QuantParams() const;
    aclnnStatus CheckFp8Params(const DataType &scaleDtype) const;
    aclnnStatus CheckFp4Params(const DataType &scaleDtype) const;
    aclnnStatus CheckNonMxQuantTransposeStatus() const;
    bool CheckTensorListSizeForEachInput() const;
    bool IsSpecialMXCase(const T *tensorList) const;
    aclnnStatus CheckMxFp8TypeKCaseInputShape(const TensorDimInfo &dimInfo, size_t index) const;
    aclnnStatus CheckMxTypeMCaseInputShape(const TensorDimInfo &dimInfo, size_t index) const;
    aclnnStatus CheckMxBiasInputShape(const TensorDimInfo &dimInfo, size_t index) const;
    bool LastTwoDimValueIsOne(const aclTensor *tensor) const;
    bool IsSpecialperTileScene(int64_t groupNum, int64_t weightNDim, int64_t weightKDim, int64_t xMDim,
                               int64_t perTokenMDim) const;

private:
    DlinferGroupedMatmulDirectParamsBase<T> gmmParams_;
    std::string xName_ = "x";
    std::string weightName_ = "weight";
    std::string scaleName_ = "scale";
    std::string perTokenScaleName_ = "perTokenScale";
    std::string groupTensorName_ = "groupTensor";
    std::string biasName_ = "bias";
    std::string yName_ = "y";
    const std::vector<op::DataType> SPECIAL_QUANT_DTYPES = {DataType::DT_FLOAT4_E1M2, DataType::DT_FLOAT4_E2M1,
                                                            DataType::DT_INT4};
};
} // namespace gmm
#endif