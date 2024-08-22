#pragma once

#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

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
    c10::string_view input_layout, int64_t num_key_value_heads);

at::Tensor npu_incre_flash_attention_v4_out(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, at::Tensor attn_output,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &quant_scale2,
    const c10::optional<at::Tensor> &quant_offset2, const c10::optional<at::Tensor> &kv_padding_size,
    int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t block_size, int64_t inner_precise);

::std::tuple<at::Tensor,at::Tensor> npu_moe_gating_topk_softmax(
    const at::Tensor &x, const at::Tensor &finishedOptional,
    int64_t topk, const at::Tensor y_out,
    const at::Tensor expert_idx_out, const at::Tensor row_idx_out);

} // namespace ascend

} // namespace dlinfer 
