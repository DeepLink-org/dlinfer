#include <torch/extension.h>

#include "moe_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // vLLM topk softmax ops
    pybind11::module ops = m.def_submodule("ops", "vLLM topk softmax operators");

    // Apply topk softmax to the gating outputs.
    ops.def("topk_softmax",
            &topk_softmax,
            "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
            "token_expert_indices, Tensor gating_output) -> ()");
}
