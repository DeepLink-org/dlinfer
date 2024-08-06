import math
import queue
import threading

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
            BaseModelOutputWithPast,
            CausalLMOutputWithPast,
            QuestionAnsweringModelOutput,
            SequenceClassifierOutputWithPast,
            TokenClassifierOutput,
)
from typing import Optional, Dict, Any, Tuple
from typing import List, Optional, Tuple, Union
import typing
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
            add_start_docstrings,
            add_start_docstrings_to_model_forward,
            is_flash_attn_greater_or_equal_2_10,
            logging,
            replace_return_docstrings,
)

import os
import sys
from .ascend_kernels import apply_rotary_pos_emb, context_attention
from dataclasses import dataclass, field, fields
import infer_ext.ops as ext_ops

def modeling_internlm2_InternLM2RMSNorm_forward(self, hidden_states):
    return ext_ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

@dataclass
class TransformerBlockContext:
    ...

transformer_block_context = TransformerBlockContext()

def modeling_internlm2_InternLM2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,  # pylint: disable=unused-argument
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        # split qkv_states by tp size
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        qkv_slices = self.wqkv.weight.split(key_value_slicing, dim=0)
        qkv_states = torch.cat(
            [F.linear(hidden_states, qkv_slice) for qkv_slice in qkv_slices], dim=-1  # pylint: disable=E1102
        )
    else:
        qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    global transformer_block_context
    if self.layer_idx == 0:
        #transformer_block_context = TransformerBlockContext()
        cos, sin = self.rotary_emb(value_states, position_ids)
        setattr(transformer_block_context, 'sin', sin)
        setattr(transformer_block_context, 'cos', cos)
    sin = transformer_block_context.sin
    cos = transformer_block_context.cos
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output = context_attention(query_states, key_states, value_states, [attention_mask.to(torch.bool)])

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.wo.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])  # pylint: disable=E1102
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def modeling_internlm2_InternLM2ForCausalLM_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    **kwargs,
):
    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
        # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
        else:
            cache_length = past_length = past_key_values[0][0].shape[1]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]  # pylint: disable=E1130

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard.
        # Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs = {"input_ids": input_ids.contiguous()}

    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
    elif use_cache:
        cache_position = cache_position[-input_length:]

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import resolve_trust_remote_code, get_class_from_dynamic_module
from transformers.utils import (
    CONFIG_NAME,
    cached_file,
    extract_commit_hash,
    find_adapter_config_file,
    is_peft_available,
)
from transformers.models.auto.configuration_auto import AutoConfig
import transformers.cache_utils


def transformers_cache_utils_dynamiccache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    else:
        #ext.ops.fill_contiguous_kvcache(self.key_cache[layer_idx], key_states)
        #ext.ops.fill_contiguous_kvcache(self.value_cache[layer_idx], value_states)
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)

    return self.key_cache[layer_idx], self.value_cache[layer_idx]


