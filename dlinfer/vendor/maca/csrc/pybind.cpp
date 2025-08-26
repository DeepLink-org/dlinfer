// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>

#include "cache.h"
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

    // Attention ops
    // Compute the attention between an input query and the cached
    // keys/values using PagedAttention.
    ops.def("paged_attention_v1",
            &paged_attention_v1,
            "paged_attention_v1("
            "    Tensor! out, Tensor query, Tensor key_cache,"
            "    Tensor value_cache, int num_kv_heads, float scale,"
            "    Tensor block_tables, Tensor seq_lens, int block_size,"
            "    int max_seq_len, Tensor? alibi_slopes,"
            "    str kv_cache_dtype, float k_scale, float v_scale,"
            "    int tp_rank, int blocksparse_local_blocks,"
            "    int blocksparse_vert_stride, int blocksparse_block_size,"
            "    int blocksparse_head_sliding_step) -> ()");

    // Rotary embedding
    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
    ops.def("rotary_embedding",
            &rotary_embedding,
            "rotary_embedding(Tensor positions, Tensor! query,"
            "                 Tensor! key, int head_size,"
            "                 Tensor cos, Tensor sin,"
            "                 bool is_neox) -> ()");

    // Cache ops
    ops.def("reshape_and_cache_new",
            &reshape_and_cache_new,
            "reshape_and_cache_new(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float kv_scale,"
            "                  float v_scale) -> ()");

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

    // Activation ops
    // Activation function used in SwiGLU.
    ops.def("silu_and_mul", &silu_and_mul, "silu_and_mul(Tensor! out, Tensor input) -> ()");

    // Layernorm
    // Apply Root Mean Square (RMS) Normalization to the input tensor.
    ops.def("rms_norm",
            &rms_norm,
            "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
            "()");

    // In-place fused Add and RMS Normalization.
    ops.def("fused_add_rms_norm",
            &fused_add_rms_norm,
            "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
            "float epsilon) -> ()");


}
