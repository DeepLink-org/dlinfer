# Copyright (c) 2024, DeepLink. All rights reserved.
import torch

from lmdeploy.pytorch.backends.default.multinomial_sampling import (
    DefaultMultinomialSamplingImpl,
)


def CambDefaultMultinomialSamplingImpl_forward(
    self,
    scores: torch.Tensor,
    seeds: torch.LongTensor,
    offsets: torch.LongTensor,
    indices: torch.Tensor = None,
):
    r"""
    Note.torch_mlu.multinomial dosen't support replacement=True, whereas lmdeploy set replacement=True by default.
    """
    sampled_index = torch.multinomial(scores, num_samples=1, replacement=False)
    outputs = torch.gather(indices, dim=1, index=sampled_index)
    return outputs.view(-1)


DefaultMultinomialSamplingImpl.forward = CambDefaultMultinomialSamplingImpl_forward
