#include "ascend_ops.hpp"
#include "op_api_common.hpp"

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
    c10::string_view input_layout, int64_t num_key_value_heads)
{
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention, query, key, value, padding_mask, atten_mask, actual_seq_lengths,
                                 num_heads, scale_value, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, attn_output);
    return attn_output;
}

} // namespace ascend

} // namespace infer_ext
