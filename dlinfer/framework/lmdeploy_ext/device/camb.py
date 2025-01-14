# Copyright (c) 2024, DeepLink. All rights reserved.
import torch

from lmdeploy.pytorch.backends.default.multinomial_sampling import (
    DefaultMultinomialSamplingImpl,
)
from lmdeploy.pytorch.engine.logits_process import (
    FusedLogitsProcessor,
    _apply_custom_logits_processors,
    _process_repetition_penalty_,
    _process_temperature_,
    _process_bad_words_,
    _guided_sampling,
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


def CambFusedLogitsProcessor__call__(
    self,
    all_ids: torch.LongTensor,
    guided_input_ids: torch.LongTensor,
    scores: torch.FloatTensor,
):
    r"""
    Note. torch_mlu.where dosen't support torch.bool,we need to convert it to torch.float16 first and then convert it to torch.bool.
    Args:
        all_ids (torch.LongTensor): All the token ids.
        guided_input_ids (torch.LongTensor): Guided prompt ids.
        scores (torch.FloatTensor):
            Prediction scores of a language modeling head.
            These can be logits for each vocabulary when not using
            beam search or log softmax for each vocabulary token
            when using beam search
    Return:
        torch.FloatTensor: The processed prediction scores.
    """
    sampling_inputs = self.sampling_inputs

    custom_logits_processors = self.sampling_inputs.logits_processors
    if any(custom_logits_processors):
        scores = _apply_custom_logits_processors(
            custom_logits_processors, all_ids, scores
        )

    repetition_penalty = sampling_inputs.repetition_penalty
    if repetition_penalty is not None:
        scores = _process_repetition_penalty_(scores, all_ids, repetition_penalty)

    temperature = sampling_inputs.temperature
    if temperature is not None:
        scores = _process_temperature_(scores, temperature)

    bad_words = sampling_inputs.bad_words
    if bad_words is not None:
        bad_mask = sampling_inputs.bad_mask
        scores = _process_bad_words_(scores, bad_words, bad_mask)

    stop_words = sampling_inputs.stop_words
    if stop_words is not None:
        stop_mask = sampling_inputs.stop_mask
        stop_mask = torch.where(
            self.ignore_eos[:, None], stop_mask.to(torch.float16), 0
        ).to(torch.bool)
        scores = _process_bad_words_(scores, stop_words, stop_mask)

    scores = _guided_sampling(
        sampling_inputs.response_formats, scores, guided_input_ids, self.tokenizer
    )
    return scores


DefaultMultinomialSamplingImpl.forward = CambDefaultMultinomialSamplingImpl_forward
FusedLogitsProcessor.__call__ = CambFusedLogitsProcessor__call__
