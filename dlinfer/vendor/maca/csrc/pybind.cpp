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

    // PagedAttention V2.
    ops.def("paged_attention_v2",
            &paged_attention_v2,
            "paged_attention_v2("
            "    Tensor! out, Tensor exp_sums, Tensor max_logits,"
            "    Tensor tmp_out, Tensor query, Tensor key_cache,"
            "    Tensor value_cache, int num_kv_heads, float scale,"
            "    Tensor block_tables, Tensor seq_lens, int block_size,"
            "    int max_seq_len, Tensor? alibi_slopes,"
            "    str kv_cache_dtype, float k_scale, float v_scale,"
            "    int tp_rank, int blocksparse_local_blocks,"
            "    int blocksparse_vert_stride, int blocksparse_block_size,"
            "    int blocksparse_head_sliding_step) -> ()");

#if 0
  ops.def(
      "page_reshape_kv_cache("
      "    Tensor key_cache, Tensor value_cache,"
      "    Tensor key_cache_new_layer, Tensor value_cache_new_layer,"
      "    int num_seqs, int num_heads, int head_size, int num_kv_heads, int block_size"      
      "	   str kv_cache_dtype) -> ()");
  //ops.def("page_reshape_kv_cache", &page_reshape_kv_cache);
  ops.impl("page_reshape_kv_cache", &page_reshape_kv_cache);
#endif

    // Activation ops
    // Activation function used in SwiGLU.
    ops.def("silu_and_mul", &silu_and_mul, "silu_and_mul(Tensor! out, Tensor input) -> ()");

    // Activation function used in GeGLU with `none` approximation.
    ops.def("gelu_and_mul", &gelu_and_mul, "gelu_and_mul(Tensor! out, Tensor input) -> ()");

    // Activation function used in GeGLU with `tanh` approximation.
    ops.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

    // GELU implementation used in GPT-2.
    ops.def("gelu_new", &gelu_new, "gelu_new(Tensor! out, Tensor input) -> ()");

    // Approximate GELU implementation.
    ops.def("gelu_fast", &gelu_fast, "gelu_fast(Tensor! out, Tensor input) -> ()");

    // Quick GELU implementation.
    ops.def("gelu_quick", &gelu_quick, "gelu_quick(Tensor! out, Tensor input) -> ()");

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

    // Cache ops
    pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");

    // Swap in (out) the cache blocks from src to dst.
    ops.def("swap_blocks", &swap_blocks, "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");

    // Copy the cache blocks from src to dst.
    ops.def("copy_blocks",
            &copy_blocks,
            "copy_blocks(Tensor[]! key_caches, Tensor[]! value_caches, Tensor "
            "block_mapping) -> ()");

    // Reshape the key and value tensors and cache them.
    ops.def("reshape_and_cache",
            &reshape_and_cache,
            "reshape_and_cache(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float k_scale, float v_scale) -> ()");

    ops.def("reshape_and_cache_new",
            &reshape_and_cache_new,
            "reshape_and_cache_new(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float kv_scale,"
            "                  float v_scale) -> ()");

    // Reshape the key and value tensors and cache them.
    ops.def("reshape_and_cache_flash",
            &reshape_and_cache_flash,
            "reshape_and_cache_flash(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype,"
            "                        float k_scale, float v_scale) -> ()");

    // Convert the key and value cache to fp8 data type.
    ops.def("convert_fp8",
            &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, str "
            "kv_cache_dtype) -> ()");

    // moe
    // Apply topk softmax to the gating outputs.
    ops.def("topk_softmax",
            &topk_softmax,
            "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
            "token_expert_indices, Tensor gating_output) -> ()");
}
