#include <torch/library.h>

#include "ascend_ops.hpp"

namespace {

TORCH_LIBRARY(npu_ext, m) {
    m.def(
        "npu_prompt_flash_attention_out(Tensor query, Tensor key, Tensor value, Tensor(a!) attn_output, *, "
        "Tensor? padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, int num_heads=1, "
        "float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, "
        "str input_layout=\"BSH\", int num_key_value_heads=0) -> Tensor(a!)");
    m.def(
        "npu_incre_flash_attention_v4_out(Tensor query, Tensor key, Tensor value, Tensor(a!) attn_output, *, "
        "Tensor? padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, "
        "Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, "
        "Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, "
        "Tensor? quant_offset2=None, Tensor? kv_padding_size=None, int num_heads=1, float scale_value=1.0, "
        "str input_layout=\"BSH\", int num_key_value_heads=0, int block_size=0, int inner_precise=1) -> Tensor(a!)");
    m.def(
        "npu_moe_gating_topk_softmax(Tensor x, Tensor? finished_opt, int topk, Tensor(a!) y_out,"
        "Tensor(b!) expert_idx_out, Tensor row_idx_out) -> (Tensor(a!), Tensor(b!))");
}

}  // namespace

namespace {

TORCH_LIBRARY_IMPL(npu_ext, PrivateUse1, m) {
    m.impl("npu_prompt_flash_attention_out", TORCH_FN(dlinfer::ascend::npu_prompt_flash_attention_out));
    m.impl("npu_incre_flash_attention_v4_out", TORCH_FN(dlinfer::ascend::npu_incre_flash_attention_v4_out));
    m.impl("npu_moe_gating_topk_softmax", TORCH_FN(dlinfer::ascend::npu_moe_gating_topk_softmax));
}

}  // namespace
