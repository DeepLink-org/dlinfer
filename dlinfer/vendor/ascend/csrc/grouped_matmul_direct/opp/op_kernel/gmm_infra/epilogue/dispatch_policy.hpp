/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
#define CATLASS_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../gmm_infra/base_defs.hpp"
#include "../../gmm_infra/arch/arch.hpp"

namespace Catlass::Epilogue {

// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};
////////////////////////////
/// new add
// For AtlasA2, GEMM
struct EpilogueAtlasA2Gemm {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, GEMV
struct EpilogueAtlasA2Gemv {
    using ArchTag = Arch::AtlasA2;
};
///////////////////////////
}  // namespace Catlass::Epilogue

#endif  // CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
