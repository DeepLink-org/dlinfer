// Copyright (c) 2024, DeepLink. All rights reserved.
#pragma once

#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

namespace dlinfer {

namespace ascend {

at::Tensor npu_prompt_flash_attention_out(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, at::Tensor& attn_output,
                                          const c10::optional<at::Tensor>& padding_mask, const c10::optional<at::Tensor>& atten_mask,
                                          c10::OptionalIntArrayRef actual_seq_lengths, int64_t num_heads, double scale_value, int64_t pre_tokens,
                                          int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads);

at::Tensor npu_incre_flash_attention_v4_out(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, at::Tensor attn_output,
                                            const c10::optional<at::Tensor>& padding_mask, const c10::optional<at::Tensor>& atten_mask,
                                            c10::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor>& antiquant_scale,
                                            const c10::optional<at::Tensor>& antiquant_offset, const c10::optional<at::Tensor>& block_table,
                                            const c10::optional<at::Tensor>& dequant_scale1, const c10::optional<at::Tensor>& quant_scale1,
                                            const c10::optional<at::Tensor>& dequant_scale2, const c10::optional<at::Tensor>& quant_scale2,
                                            const c10::optional<at::Tensor>& quant_offset2, const c10::optional<at::Tensor>& kv_padding_size, int64_t num_heads,
                                            double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size,
                                            int64_t inner_precise);

at::Tensor npu_weight_quant_batchmatmul_out(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& antiquant_scale,
                                            const at::optional<at::Tensor>& antiquant_offset, const at::optional<at::Tensor>& quant_scale,
                                            const at::optional<at::Tensor>& quant_offset, const at::optional<at::Tensor>& bias, int64_t group_size,
                                            at::Tensor matmul_output);

::std::tuple<at::Tensor, at::Tensor> npu_moe_gating_topk_softmax(const at::Tensor& x, const c10::optional<at::Tensor>& finished_opt, int64_t topk,
                                                                 at::Tensor& y_out, at::Tensor& expert_idx_out, at::Tensor& row_idx_out);

}  // namespace ascend

}  // namespace dlinfer
