// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>

#include "moe/moe_ops.h"
#include "ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // vLLM custom ops
    pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

    // Rotary embedding
    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
    ops.def("rotary_embedding",
            &rotary_embedding,
            "rotary_embedding(Tensor positions, Tensor! query,"
            "                 Tensor! key, int head_size,"
            "                 Tensor cos, Tensor sin,"
            "                 bool is_neox) -> ()");

    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
    // (supports multiple loras).
    ops.def("batched_rotary_embedding",
            &batched_rotary_embedding,
            "batched_rotary_embedding(Tensor positions, Tensor! query,"
            "                         Tensor! key, int head_size,"
            "                         Tensor cos_sin_cache, bool is_neox,"
            "                         int rot_dim,"
            "                         Tensor cos_sin_cache_offsets) -> ()");

    // Aligning the number of tokens to be processed by each expert such
    // that it is divisible by the block size.
    ops.def("moe_align_block_size",
            &moe_align_block_size,
            "moe_align_block_size(Tensor topk_ids, int num_experts,"
            "                     int block_size, Tensor! sorted_token_ids,"
            "                     Tensor! experts_ids,"
            "                     Tensor! num_tokens_post_pad) -> ()");

    // moe
    // Apply topk softmax to the gating outputs.
    ops.def("topk_softmax",
            &topk_softmax,
            "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
            "token_expert_indices, Tensor gating_output) -> ()");
}
