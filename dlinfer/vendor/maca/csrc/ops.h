// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include <torch/library.h>

#include <optional>

void paged_attention_v1(torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
                        torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size, torch::Tensor& max_seq_len,
                        const c10::optional<torch::Tensor>& alibi_slopes, const std::string& kv_cache_dtype, double k_scale, double v_scale,
                        const int64_t tp_rank, const int64_t blocksparse_local_blocks, const int64_t blocksparse_vert_stride,
                        const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out, torch::Tensor& query,
                        torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale, torch::Tensor& block_tables,
                        torch::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
                        const std::string& kv_cache_dtype, double k_scale, double v_scale, const int64_t tp_rank, const int64_t blocksparse_local_blocks,
                        const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step);

#if 0
void page_reshape_kv_cache(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& key_cache_new_layer,
    torch::Tensor& value_cache_new_layer,
    int64_t num_seqs,
    int64_t num_heads,
    int64_t head_size,
    int64_t num_kv_heads,
    int64_t block_size,
    const std::string& kv_cache_dtype);
#endif

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size, torch::Tensor& cos, torch::Tensor& sin,
                      bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim, torch::Tensor& cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void advance_step(int64_t num_seqs, int64_t num_queries, int64_t block_size, torch::Tensor& input_tokens, torch::Tensor& sampled_token_ids,
                  torch::Tensor& input_positions, torch::Tensor& seq_lens, torch::Tensor& slot_mapping, torch::Tensor& block_tables);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size, torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);
