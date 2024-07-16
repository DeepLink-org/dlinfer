#include <torch/library.h>
#include "ascend_ops.hpp"

namespace {

TORCH_LIBRARY(npu_ext, m) {
    m.def("npu_prompt_flash_attention_out(Tensor query, Tensor key, Tensor value, Tensor(a!) attn_output, *, "
          "Tensor? padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, int num_heads=1, "
          "float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, "
          "str input_layout=\"BSH\", int num_key_value_heads=0) -> Tensor(a!)");
}

} // namespace

namespace {

TORCH_LIBRARY_IMPL(npu_ext, PrivateUse1, m) {
    m.impl("npu_prompt_flash_attention_out", TORCH_FN(infer_ext::ascend::npu_prompt_flash_attention_out));
}

} // namespace
