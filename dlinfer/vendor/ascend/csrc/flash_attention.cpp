// Copyright (c) 2024, DeepLink. All rights reserved.
#include "ascend_ops.hpp"
#include "op_api_common.hpp"
#include <ATen/core/ATen_fwd.h>
#include <c10/util/OptionalArrayRef.h>

namespace dlinfer {

namespace ascend {

at::Tensor npu_prompt_flash_attention_out(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, at::Tensor &attn_output,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths,
    int64_t num_heads, double scale_value,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads)
{
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    at::IntArrayRef actual_seq_len = actual_seq_lengths.value_or(at::IntArrayRef{});

    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention, query, key, value, padding_mask, atten_mask, actual_seq_len,
        num_heads, scale_value, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, attn_output);
    return attn_output;
}

at::Tensor npu_incre_flash_attention_v4_out(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, at::Tensor attn_output,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &quant_scale2,
    const c10::optional<at::Tensor> &quant_offset2, const c10::optional<at::Tensor> &kv_padding_size,
    int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t block_size, int64_t inner_precise)
{
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    at::TensorList key_cache = key;
    at::TensorList value_cache = value;
    at::IntArrayRef actual_seq_len = actual_seq_lengths.value_or(at::IntArrayRef{});

    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttentionV4, query, key_cache, value_cache, padding_mask, atten_mask, actual_seq_len,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table,
        kv_padding_size, num_heads, scale_value, input_layout_ptr, num_key_value_heads, block_size, inner_precise, attn_output);
    return attn_output;
}

} // namespace ascend

} // namespace dlinfer
