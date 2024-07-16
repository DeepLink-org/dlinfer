#pragma once

#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

namespace infer_ext {

namespace ascend {

at::Tensor npu_prompt_flash_attention_out(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, at::Tensor &attn_output,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    c10::optional<at::IntArrayRef> actual_seq_lengths,
    int64_t num_heads, double scale_value,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads);

} // namespace ascend

} // namespace infer_ext 
