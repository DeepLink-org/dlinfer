/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_DISPATCH_POLICY_HPP
#define CATLASS_GEMM_DISPATCH_POLICY_HPP

#include "../../gmm_infra/base_defs.hpp"
#include "../../gmm_infra/arch/arch.hpp"

namespace Catlass::Gemm {

// Block Mmad Policies

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;
using MmadAtlasA2Async = MmadAtlasA2Base<true>;

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L1A_STAGES_, uint32_t L1A_TILE_NUM_, uint32_t L1B_STAGES_,
    uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_RIFFLE_SHUFFLE_>
struct MmadAtlasA2PreloadAsyncFixAxisMoveWithCallback : public MmadAtlasA2Async {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;  // Stages of emitting load instruction in advance
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1A_TILE_NUM = L1A_TILE_NUM_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_RIFFLE_SHUFFLE = ENABLE_RIFFLE_SHUFFLE_;
};


////////////////////
// new add
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_RIFFLE_SHUFFLE_ = false, bool ENABLE_ABBA_ = false>
struct GemmAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_RIFFLE_SHUFFLE = ENABLE_RIFFLE_SHUFFLE_;
    static constexpr bool ENABLE_ABBA = ENABLE_ABBA_;
};

struct GemvAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
////////////////////

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongBias : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
}  // namespace Catlass::Gemm

#endif  // CATLASS_GEMM_DISPATCH_POLICY_HPP
