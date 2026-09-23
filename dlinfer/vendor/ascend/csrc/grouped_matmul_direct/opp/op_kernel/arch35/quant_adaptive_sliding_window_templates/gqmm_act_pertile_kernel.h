/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gqmm_act_pertile_kernel.h
 * \brief
 */
#ifndef GQMM_ACT_PERTILE_KERNEL_H
#define GQMM_ACT_PERTILE_KERNEL_H

#include "../../../../common/groupedmatmul_act/epilogue/block_epilogue_pertile.h"
#include "../../../../common/groupedmatmul_act/block/block_mmad_pertile.h"
#include "../../../../common/groupedmatmul_act/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../../../../common/groupedmatmul_act/block/block_scheduler_policy.h"
#include "../../../../common/groupedmatmul_act/kernel/kernel_qgmm_pertile.h"
#include "../../../../common/groupedmatmul_act/policy/dispatch_policy.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"
#include "quant_utils.h"
using GMMQuantParams = DlinferGroupedMatmulDirectTilingData::GMMQuantParams;

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmActPerTileKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                           GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                           const GMMQuantParams *gmmBaseParamsIn, const TCubeTiling *mmTilingDataIn,
                                           AscendC::TPipe *que)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<Act::Gemm::_0, Act::Gemm::_0, Act::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Act::Gemm::_0, Act::Gemm::_0, Act::Gemm::_0>;

    // 定义矩阵的类型和布局
    using AType = xType;
    using BType = wType;
    using CType = l0cType;
    using BiasType = biasType;
    using ScaleType = scaleType;
    using PtScaleType = ptScaleType;
    using YType = yType;

    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using LayoutY = yLayout;
    using LayoutBias = yLayout;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = Act::Gemm::MatmulShape;

    // 定义scheduler类型
    using BlockScheduler = Act::Gemm::DlinferGroupedMatmulDirectAswtWithTailSplitScheduler;

    // 定义MMAD类型
    using BlockMmadPolicy = Act::Gemm::GMMPerTile<>;
    using BlockMmad = Act::Gemm::Block::BlockMmadGmm<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                     BiasType, LayoutBias, L1TileShape, L0TileShape>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Act::Gemm::Block::BlockEpiloguePerTile<L0TileShape, YType, CType, BiasType, ScaleType,
                                                                 LayoutA, PtScaleType, LayoutB>;

    // 定义Kernel类型
    using GmmKernel = Act::Gemm::Kernel::QuantMmGroupedPerTile<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;
    GMMTiling gmmParams{mmTilingDataIn->M,
                        mmTilingDataIn->N,
                        mmTilingDataIn->Ka,
                        mmTilingDataIn->baseM,
                        mmTilingDataIn->baseN,
                        mmTilingDataIn->baseK,
                        mmTilingDataIn->stepM,
                        mmTilingDataIn->stepN,
                        mmTilingDataIn->stepKa,
                        mmTilingDataIn->stepKb,
                        gmmBaseParamsIn->groupNum,
                        gmmBaseParamsIn->groupType,
                        gmmBaseParamsIn->groupListType};
    // G-B only support S-S-S
    Params params = {{1, 1, 1, 1},                    // shape
                     {x, weight, y, bias, groupList}, // gm addr
                     {(GM_ADDR)GROUPED_MATMUL::GetTensorAddr<YType>(0, y),
                      (GM_ADDR)GROUPED_MATMUL::GetTensorAddr<ScaleType>(0, scale), perTokenScale, nullptr,
                      static_cast<uint32_t>(mmTilingDataIn->baseM), static_cast<uint32_t>(mmTilingDataIn->baseN),
                      static_cast<uint32_t>(mmTilingDataIn->baseK), 1, QuantUtils::PER_BLOCK_SIZE,
                      QuantUtils::PER_BLOCK_SIZE}, // epilogue args
                     gmmParams};
    GmmKernel gmm;
    gmm(params);
}
#endif