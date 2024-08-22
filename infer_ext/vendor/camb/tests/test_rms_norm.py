import torch
import torch_mlu
from bangtransformer.torch import bt_ops
from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from torch.nn.parameter import Parameter

import sys
sys.path.append("..")
import camb_ops

class FuseRMSNorm(torch.nn.Module):
    def __init__(self, weights, eps=1e-6):
        super().__init__()

        self.weight = weights
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual

def compare_tensor(a, b, prec):
    epsilon = 1.0 / 16384

    diff = a - b
    diff = diff.abs().pow(2).sum()
    a_pow_sum = a.pow(2).sum()
    if diff <= (2 * epsilon) * (2 * epsilon):
        diff = 0.0
    if a_pow_sum <= epsilon:
        a_pow_sum = a_pow_sum + epsilon
    diff = torch.div(diff, (a_pow_sum * 1.0))
    return diff.sqrt().item() <= prec

def test_rms_norm0():
    H, C = 4096, 4096
    eps = 1e-6
    input = torch.randn(H, C, device="mlu", dtype=torch.half)
    residual = torch.randn(H, C, device="mlu", dtype=torch.half)
    weight = Parameter(torch.randn(C, device="mlu", dtype=torch.half))
    
    print("test_rms_norm0: H={}, C={}, testing...".format(H, C))
    ref_normed_out = vendor_ops_registry["rms_norm"](input, weight, eps)
    rms_norm = FuseRMSNorm(weight, eps=eps)
    normed_out = rms_norm(input, None)[0]

    if compare_tensor(normed_out, ref_normed_out, 0.003):
        print("test_rms_norm0: pass")
    else:
        print(f"test_rms_norm0: not close, max diff: {max_diff}")

def test_rms_norm1():
    N, H, C = 64, 64, 4096
    eps = 1e-6
    input = torch.randn(N, H, C, device="mlu", dtype=torch.half)
    residual = torch.randn(N, H, C, device="mlu", dtype=torch.half)
    weight = Parameter(torch.randn(C, device="mlu", dtype=torch.half))
   
    print("test_rms_norm1: N={}, H={}, C={}, testing...".format(N, H, C))
    ref_normed_out = vendor_ops_registry["rms_norm"](input.clone(), weight, eps)
    rms_norm = FuseRMSNorm(weight, eps=eps)
    normed_out = rms_norm(input, None)[0]

    if compare_tensor(normed_out, ref_normed_out, 0.003):
        print("test_rms_norm1: pass")
    else:
        print(f"test_rms_norm1: not close, max diff: {max_diff}")
        
if __name__ == "__main__":
    test_rms_norm0()
    test_rms_norm1()
