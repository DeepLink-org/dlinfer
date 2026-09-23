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
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_utils.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = DlinferGroupedMatmulDirectTilingData::GMMWeightQuantTilingData;
using GMMNoQuantTilingData = DlinferGroupedMatmulDirectTilingData::GMMNoQuantTilingData;
using GMMQuantTilingData = DlinferGroupedMatmulDirectTilingData::GMMQuantTilingData;
#if defined(V310_GMM_QUANT)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif
#if defined(V310_GMM_QUANT_PERTILE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_act_pertile_kernel.h"
#endif
#elif defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::A16MXF4_NZKN;
using WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK;
using WeightQuantBatchMatmulV2::Arch35::S8S4_NZKN_G;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlock;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {3, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_5 = {3, 384};
#if defined(DT_FLOAT) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#if defined(DT_INT32) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_INT32
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT AscendC::int4b_t
    #undef ORIG_DTYPE_WEIGHT
    #define ORIG_DTYPE_WEIGHT DT_INT4
#endif
#else
#include "arch35/non_quant/grouped_matmul_basic_kernel.h"
#include "arch35/non_quant/grouped_matmul_tiling_key.h"
#endif
#else
#include "grouped_matmul_antiquant.h"
#include "grouped_matmul_vector.h"
#include "grouped_matmul_tiling_key.h"
#include "grouped_matmul.h"
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)

#include "grouped_matmul_antiquant_a16w8_msd.h"
#include "grouped_matmul_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_antiquant_a8w4_msd.h"
#include "grouped_matmul_antiquant_a8w4_pre.h"
#include "grouped_matmul_antiquant_a8w4.h"
#include "grouped_matmul_antiquant_a8w4_msd_new.h"
#include "grouped_matmul_quant_mixcore.h"
#include "grouped_matmul_pre_tiling.h"
#include "grouped_matmul_a4w4.h"
#include "grouped_matmul_autotiling_a8w4.h"
#ifndef __CCE_KT_TEST__
#include "grouped_matmul_fixaxismove_interface.cpp"
#endif
#endif


using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
constexpr MatmulConfig matmulCFG = NZ_CFG_MDL;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
constexpr MatmulConfig matmulCFG = CFG_MDL;
#endif

#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
constexpr auto GetMmCFG() {
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using xTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

template <bool trans = false>
using weightTypeMSD = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_WEIGHT, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using yTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

namespace {
    __aicore__ inline static constexpr MatmulApiStaticTiling GetGmmMatmulApiTiling(bool isND2NZ, bool transB) {
        MatmulConfig conf = GenGmmConf(isND2NZ);
        MatmulApiStaticTiling staticTilingTmp;
        if (transB) {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<true>, yType, biasType>(conf);
        } else {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<false>, yType, biasType>(conf);
        }
        staticTilingTmp.depthA1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.depthB1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.stepM = 1;
        staticTilingTmp.stepN = 1;
        staticTilingTmp.stepKa = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.stepKb = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.dbL0A = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0B = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0C = 1;
        return staticTilingTmp;
    }
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
    constexpr bool isWeightNZ = true;
#else
    constexpr bool isWeightNZ = false;
#endif
    constexpr static auto staticCFG = GetGmmMatmulApiTiling(isWeightNZ, false);
    constexpr static auto staticCFGtransB = GetGmmMatmulApiTiling(isWeightNZ, true);
} // namespace


#define GMM_IMP(computeClass, processClass, transA, transB, sync, cfg)                                             \
    do {                                                                                                           \
        using matmulType = MMType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                        \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);                                       \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_STATIC_TILING_IMP(processClass, transA, transB, sync, cfg)                                        \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                    \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, nullptr, &tPipe);                                               \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_STATIC_TILING_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType) \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, nullptr, &tPipe);                                                \
        computeOp.InitStaticTiling(&gmmBaseParams_, user1, (cfg).baseM, (cfg).baseN);                              \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, &mmTilingData_, &tPipe);                                        \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#if defined(CONST_TILING)
#define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)               \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GMMTilingData gmmTilingData;                                                                               \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#else
    #define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)           \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        GMMPreTilingProcess preTiling;                                                                             \
        preTiling.Init(groupList, gmmBaseParams_, mmTilingData_, &tPipe);                                          \
        preTiling.Process(gmmBaseParams_, mmTilingData_);                                                          \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#endif

#define GMM_A4W4_IMP(computeClass, transA, transB, cfg, aType, bType, cType)                                       \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> computeOp(mm);                                                                    \
        computeOp.Init(x, weight, scale, groupList, perTokenScale,                                                 \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                            \
        computeOp.Process();                                                                                       \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg)                                                               \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4PreProcess op1;                                                                                 \
            op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe);                                              \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>;                          \
        using bT = MatmulType<TPosition::GM, wFormat, DTYPE_WEIGHT_DEV_A8W4MSD, false>;                            \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, offset, nullptr, nullptr, nullptr,               \
                y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4(computeClass, cfg)                                                                   \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, groupList, user1, gmmBaseParams_, &tPipe);                                         \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, offset, nullptr, nullptr, nullptr,               \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                            \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(computeClass, cfg)                                                          \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, scale, user1, gmmBaseParams_, &tPipe);                                             \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        GlobalTensor<int8_t> yGm;                                                                                  \
        yGm.SetGlobalBuffer((__gm__ int8_t *)workspace);                                                           \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                      \
        using matmulType = MMImplType<aT, bT, cT, biasT, matmulCFG>;                                               \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        GMMQuantMixCoreCompute<matmulType, false> computeOp(mm);                                                   \
        computeOp.isA8W4FakeQuant = true;                                                                          \
        computeOp.Init(x, user1, bias, user1, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,   \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        GMMProcess<decltype(computeOp)> op(computeOp);                                                             \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        GET_TILING_DATA_MEMBER_ADDR(GMMWeightQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                        \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                     \
        op.Init(x, weight, antiquantScale, antiquantOffset, bias, groupList, y, &gmmBaseParams_,                       \
                &mmTilingData_, tiling, gmmArrayAddr_, &tPipe);                                                        \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                         \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);           \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, float,                          \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                         \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(templateClass, ...)                                    \
    do {                                                                                                         \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);           \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE,          \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                         \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(templateClass, ...)                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE,                \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantVcvMatmulBasicBlock, __VA_ARGS__> op;                            \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                                  \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                                \
                      transposeX1, transposeX2>                                                                        \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                              \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, wFormat,                         \
                      transposeX1, transposeX2, DTYPE_L0C_LOCAL>                                                       \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        if ASCEND_IS_AIC {                                                                                             \
            templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                            \
                          transposeX1, transposeX2>                                                                    \
                op;                                                                                                    \
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,      \
                    gmmArrayAddr_, &tPipe);                                                                            \
            op.Process();                                                                                              \
        }                                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            GQmmEmptyTensor<DTYPE_Y>(groupList, y, &gmmQuantParams_, gmmArrayAddr_, mmTilingData_.usedCoreNum,         \
                                     &tPipe);                                                                          \
        }                                                                                                              \
    } while (0)

#define GMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout)                                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GmmActPerTileKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                            DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1,               \
                                             &gmmQuantParams_, &mmTilingData_, &tPipe);                                \
    } while (0)

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#if defined(V310_GMM_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#elif defined(V310_GMM_ANTI_QUANT)
template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
#else
template <int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
__global__ __aicore__ void dlinfer_grouped_matmul_direct(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
#else
template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE,
          int IS_STATIC_TILING_API, int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO, bool IS_ENABLE_FIXED_AXIS>
__global__ __aicore__ void dlinfer_grouped_matmul_direct(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
#endif
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT) // Quant: A8W8
REGISTER_TILING_DEFAULT(GMMQuantTilingData);
#if defined(V310_GMM_QUANT_MX) // mxfpx
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // scale64/perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // mx/perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(true, false, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MIX) // perToken/SPLIT_K/scale bf16/fp32
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, false, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, true, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(true, false, GQmmMixRegbaseKernel);
    }
#endif
#if defined(V310_GMM_QUANT_PERTILE)
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajor,
                                Act::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::ColumnMajor,
                                Act::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajor,
                                Act::Gemm::layout::RowMajorAlign);
    }
#endif
#elif defined(V310_GMM_ANTI_QUANT)
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X == DT_INT8
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_4);
        } else if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
                             OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_384_BUF_NUM_3 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_5);
        }
    #elif ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, MXA8W4_NZNK,
                                                              VEC_ANTIQUANT_CONFIG_3);
        }
    #elif ORIG_DTYPE_ANTIQUANT_SCALE == DT_FLOAT8_E8M0
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, A16MXF4_NZKN,
                                                               VEC_ANTIQUANT_CONFIG_3);
        }
    #else
        if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,
                                                QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        }
    #endif
#else
    REGISTER_TILING_DEFAULT(GMMNoQuantTilingData);
    if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_TRANS) {    // x transposed
        if ASCEND_IS_AIV {
            EmptyTensor<DTYPE_Y>(groupList, y, tiling);
        }
        if ASCEND_IS_AIC {
            if constexpr (wFormat == CubeFormat::NZ) {
                GmmNoQuantAswt<layout::ColumnMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
            } else {
                GmmNoQuantAswt<layout::ColumnMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
            }
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {    // weight transposed
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Zn>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::ColumnMajor>(x, weight, bias, groupList, y, tiling);
        }
    }
#endif
#endif
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
    // ANTIQUANT_A8W4
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT4) {
        if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT) {
            GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDCompute, A8W4_GMM_CFG_MDL);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT) {
            GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDComputeNew, A8W4_GMM_CFG_MDL_NEW);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT) {
            GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT) {
            GMM_CV_SPLIT_IMP_A8W4(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING) {
            GET_TILING_DATA_MEMBER(GMMTilingData, hpTilingData, tilingData, tiling);
            GM_ADDR A = x;
            GM_ADDR B = weight;
            GM_ADDR C = y;
            GM_ADDR groupListOptional = groupList;
            GM_ADDR bias_ = bias;
            GM_ADDR offset_ = offset;
            GM_ADDR sa = perTokenScale;
            GM_ADDR sw = scale;
            GM_ADDR workspaceDevice = user1;

            GMMA4W8AutotilingCompute op(A, B, C, groupListOptional, bias_, offset_, sa, sw, workspaceDevice,
                                        const_cast<A8W4HPTiling *>(&tilingData), &tPipe);
            op.Init();
            op.Process();
        }
    }
#elif defined(GMM_ANTI_QUANT)
    // ANTIQUANT
    if constexpr ((D_T_A == GMM_TPL_FLOAT16 || D_T_A == GMM_TPL_BF16) &&
                  A16W8_KERNEL_TEMPLATE != GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD) {
        // ANTIQUANT_A16W4 & ANTIQUANT_A16W8_NOT_MSD
        if constexpr (TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
            GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, false, false, matmulCFG);
        } else if constexpr (TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
            GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, true, false, matmulCFG);
        } else if constexpr (TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_2) {
            GMM_IMP(GMMAntiquantComputePerformance, GMMAntiquantProcess, false, false, false, matmulCFG);
        }
    }
#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT8) && ORIG_DTYPE_WEIGHT == DT_INT8
    // ANTIQUANT_A16W8_MSD
    if constexpr ((D_T_A == GMM_TPL_FLOAT16 || D_T_A == GMM_TPL_BF16) && D_T_B == GMM_TPL_INT8 &&
                  A16W8_KERNEL_TEMPLATE == GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD) {
        if constexpr (TRANS_B == 0) {
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, false, false, matmulCFG,
                             xTypeMSD, weightTypeMSD, yTypeMSD);
        } else if constexpr (TRANS_B == 1) {
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, true, false, matmulCFG,
                             xTypeMSD, weightTypeMSD, yTypeMSD);
        }
    }
#endif

#elif defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16)
    // QUANT_A8W8O16
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT8 && (D_T_Y == GMM_TPL_BF16 || D_T_Y == GMM_TPL_FLOAT16) &&
                  TRANS_A == 0 && A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
        if constexpr (IS_STATIC_TILING_API == 0) {
            if constexpr (AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
                if constexpr(IS_ENABLE_FIXED_AXIS == 0) {
                    if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
                    } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
                    } else if constexpr(TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, false, false, matmulCFG, xType,
                                    weightType, yType);
                    } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, true, false, matmulCFG, xType,
                                    weightType, yType);
                    }
                } else if constexpr(IS_ENABLE_FIXED_AXIS == 1 && TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM) {
                    tPipe.Destroy();
                    AscendC::SetMMLayoutTransform(true);
                    GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling)
                    using XDType = int8_t;
                    using WeightDType = int8_t;
                    using CDType = int32_t;
                    using ScaleDType = float;
                    using GrouplistDType = int64_t;
                    using PerTokenScaleDType = float;
                    using YDType = half;
#ifndef __CCE_KT_TEST__
                    Catlass::grouped_matmul_fixaxismove<XDType, WeightDType, CDType, ScaleDType, GrouplistDType, PerTokenScaleDType, YDType>(
                        gmmBaseParams_.m, gmmBaseParams_.k, gmmBaseParams_.n, gmmBaseParams_.groupNum,
                        x, weight, scale, groupList, perTokenScale, y, user1, gmmBaseParams_.coreNum);
#endif
                }
            } else if constexpr (AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_2) {
                if constexpr (TRANS_B == 0) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
                } else if constexpr (TRANS_B == 1) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
                }
            }
        } else if (IS_STATIC_TILING_API == 1) {
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                            false, false, false, staticCFG, xType, weightType, yType);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                            false, true, false, staticCFGtransB, xType, weightType, yType);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                            false, false, false, staticCFG, xType, weightType, yType);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                            false, true, false, staticCFGtransB, xType, weightType, yType);
            }
        }
    }
#elif defined(GMM_A4W4)
    // QUANT_A4W4
    if constexpr (D_T_A == GMM_TPL_INT4 && D_T_B == GMM_TPL_INT4) {
        GMM_A4W4_IMP(GMMA4W4Compute, false, false, matmulCFG, xType, weightType, yType);
    }
#elif defined(GMM_QUANT_INT8) || defined(GMM_QUANT_INT32)
    // QUANT_A8W8O8 & QUANT_A8W8O32
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT8 && (D_T_Y == GMM_TPL_INT8 || D_T_Y == GMM_TPL_INT32) &&
                  TRANS_A == 0 && A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE &&
                  AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
        if constexpr (IS_STATIC_TILING_API == 0) {
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMGroupMSparseProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMGroupMSparseProcess, false, true, false, matmulCFGUnitFlag);
            }
        } else if constexpr (IS_STATIC_TILING_API == 1){
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, false, false, staticCFG);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, true, false, staticCFGtransB);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, false, false, staticCFG);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, true, false, staticCFGtransB);
            }
        }
    }
#elif defined(GMM_FLOAT)
    // NO_QUANT
    if (IS_STATIC_TILING_API == 0 &&
        A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
            if constexpr (TRANS_A == 0 && TRANS_B == 0 &&
                          GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
                          AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 0 && TRANS_B == 1 &&
                                 GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
                                 AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 0 && TRANS_B == 0 &&
                                 GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
                                 AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMGroupMSparseProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 0 && TRANS_B == 1 &&
                                 GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
                                 AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMGroupMSparseProcess, false, true, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
                if ASCEND_IS_AIV {
                    GET_TILING_DATA(tilingData, tiling);
                    EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
                }
                if ASCEND_IS_AIC {
                    GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
                }
            }
    }

#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
#if defined(GMM_FLOAT)
    if constexpr (TRANS_A == 0 && TRANS_B == 0) {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFG);
    } else if constexpr (TRANS_A == 0 && TRANS_B == 1) {
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFG);
    } else if constexpr (TRANS_A == 1 && TRANS_B == 0) {
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
        }
    }
#endif
#endif
}
