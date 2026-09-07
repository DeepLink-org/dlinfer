# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
import torch
from functools import lru_cache
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.strategies.base.model_agent import ExtraInputs
from dlinfer.vendor import vendor_name

vendor = ["camb", "ascend"]


def fake_torch_compile(dynamic=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


def post_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    variance: torch.Tensor,
    eps: float,
    embed_dim: int,
    dtype: torch.dtype,
):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


def patch_compiled_func():
    import torch

    real_torch_compile = torch.compile
    torch.compile = fake_torch_compile
    from lmdeploy.pytorch.models import internvl, internvl3_hf

    internvl.pre_rms_norm = pre_rms_norm
    internvl.post_rms_norm = post_rms_norm
    internvl3_hf.pre_rms_norm = pre_rms_norm
    internvl3_hf.post_rms_norm = post_rms_norm
    torch.compile = real_torch_compile


def patch_async_sampling_logits():
    from torch.profiler import record_function
    from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
    from lmdeploy.pytorch.strategies.base.model_agent import ExtraInputs
    from lmdeploy.pytorch.model_inputs import ModelInputs
    from lmdeploy.pytorch.engine.logits_process import (
        SamplingInputs,
        FusedLogitsProcessor,
    )

    async def async_sampling_logits(
        self,
        logits: torch.Tensor,
        inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
    ):
        """Sampling logits."""
        if self.spec_agent.is_enabled():
            extra_inputs.target_logits = extra_inputs.target_logits.to(torch.float32)
            extra_inputs = await self.spec_agent.async_sampling_logits(
                inputs, extra_inputs, sampling_inputs
            )
            return (
                extra_inputs.next_token_ids,
                extra_inputs.logprobs,
                extra_inputs.output_token_ids,
                extra_inputs,
            )
        # record function does not support async function
        # so we can not decorate it on async_sampling_logits
        with record_function("sampling_logits"):
            logits = logits.to(torch.float32)
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=self.guided_decoding_manager,
            )
            origin_logits = logits
            logits, raw_logprobs = await logits_processor(origin_logits)
            next_token_ids = logits_processor.sampling(logits)
            await logits_processor.accept_guided_tokens(next_token_ids)
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )
        # post sampling
        next_token_ids, extra_inputs = self.agent_strategy.post_sampling(
            inputs, logits, next_token_ids, extra_inputs
        )
        return next_token_ids, logprobs, next_token_ids, extra_inputs

    BaseModelAgent.async_sampling_logits = async_sampling_logits


def patch_rejection_sampler():
    from lmdeploy.pytorch.spec_decode import reject_sampler as _reject_sampler_mod
    from dlinfer.vendor.ascend.triton_ops.reject_sample import rejection_sample

    def _patched_rejection_sample(
        target_logits,
        draft_token_ids,
        bonus_token_ids,
        sampling_inputs,
        draft_probs=None,
    ):
        if not target_logits.is_contiguous():
            target_logits = target_logits.contiguous()
        if not draft_token_ids.is_contiguous():
            draft_token_ids = draft_token_ids.contiguous()
        if draft_probs is not None and not draft_probs.is_contiguous():
            draft_probs = draft_probs.contiguous()

        # origin target_logits is torch.bfloat16
        target_logits = target_logits.to(torch.float32)
        return rejection_sample(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            sampling_inputs,
            draft_probs=draft_probs,
        )

    _reject_sampler_mod.rejection_sample = _patched_rejection_sample


def patch_modelslim_quantization_config():
    """Add Ascend ModelSlim dispatch to LMDeploy's quantization config."""
    from collections.abc import Mapping

    from lmdeploy.pytorch.config import QuantizationConfig

    if getattr(QuantizationConfig, '_dlinfer_modelslim_patched', False):
        return

    original_from_config = QuantizationConfig.from_config
    original_get_quant_method = QuantizationConfig.get_quant_method

    @classmethod
    def custom_from_config(cls, hf_config):
        quant_sources = []
        quant_config = getattr(hf_config, 'quantization_config', None)
        if quant_config is not None:
            quant_sources.append(quant_config)
        for config_name in ('llm_config', 'text_config'):
            nested_config = getattr(hf_config, config_name, None)
            nested_quant_config = getattr(nested_config,
                                          'quantization_config', None)
            if nested_quant_config is not None:
                quant_sources.append(nested_quant_config)

        if not quant_sources:
            return original_from_config(hf_config)
        if any(
                isinstance(config, Mapping)
                and config.get('quant_method') == 'compressed-tensors'
                for config in quant_sources):
            return original_from_config(hf_config)

        quant_config = quant_sources[0]
        if (not isinstance(quant_config, Mapping)
                or quant_config.get('quant_method') != 'modelslim'):
            return original_from_config(hf_config)

        quant_dtype = quant_config.get('quant_dtype') or 'int8'
        resolved_quant_dtype = getattr(torch, quant_dtype, None)
        if not isinstance(resolved_quant_dtype, torch.dtype):
            raise ValueError(
                f'Invalid quant dtype "{quant_dtype}" resolved from model '
                'config; expected a torch.dtype attribute on torch.')

        ignored_layers = quant_config.get('ignored_layers', [])
        if not ignored_layers:
            ignored_layers = quant_config.get('modules_to_not_convert', [])
        return cls(
            quant_method='modelslim',
            quant_dtype=resolved_quant_dtype,
            scale_fmt=quant_config.get('scale_fmt'),
            weight_block_size=quant_config.get('weight_block_size'),
            activation_scheme=quant_config.get('activation_scheme'),
            ignored_layers=ignored_layers,
            fp8_quant_scope=quant_config.get('fp8_quant_scope'),
            hf_quant_config=quant_config,
        )

    def get_modelslim_quant_method(self, prefix, module_kind):
        if not prefix or module_kind == 'norm':
            return None

        description = self.hf_quant_config.get('quant_description', {})
        if not description:
            raise ValueError(
                'ModelSlim quantization requires quant_description metadata.')

        proj_name = prefix.rsplit('.', 1)[-1]
        if module_kind == 'moe':
            suffixes = ('0.gate_proj.weight', '0.up_proj.weight',
                        '0.down_proj.weight')
            keys = [f'{prefix}.{suffix}' for suffix in suffixes]
        elif proj_name == 'gate_up_proj':
            parent = prefix.rsplit('.', 1)[0]
            keys = [f'{parent}.gate_proj.weight',
                    f'{parent}.up_proj.weight']
        else:
            keys = [f'{prefix}.weight']

        missing = [key for key in keys if key not in description]
        if missing:
            return None
        quant_types = {description[key] for key in keys}
        if len(quant_types) != 1:
            raise ValueError(
                f'ModelSlim fused module {prefix} mixes quant types: '
                f'{sorted(quant_types)}')

        quant_type = quant_types.pop()
        if quant_type == 'FLOAT':
            return None
        if quant_type == 'W8A8_DYNAMIC':
            return 'smooth_quant'
        if quant_type == 'W8A8':
            if module_kind == 'moe':
                raise ValueError(
                    f'Static W8A8 MoE is not supported for {prefix}.')
            return 'modelslim_w8a8_static'
        raise ValueError(
            f'Unsupported ModelSlim quant type {quant_type!r} for {prefix}.')

    def custom_get_quant_method(self,
                                prefix='',
                                module_kind='linear'):
        if self.quant_method != 'modelslim':
            return original_get_quant_method(self, prefix, module_kind)
        if module_kind not in {'linear', 'moe', 'norm'}:
            raise ValueError(
                f'Unsupported quant module kind: {module_kind}')
        return self._get_modelslim_quant_method(prefix, module_kind)

    QuantizationConfig.from_config = custom_from_config
    QuantizationConfig._get_modelslim_quant_method = (
        get_modelslim_quant_method)
    QuantizationConfig.get_quant_method = custom_get_quant_method
    QuantizationConfig._dlinfer_modelslim_patched = True


def patch_deepseek_v32_config():
    """Allow the DeepSeek-V3.2 config builder to run on Ascend.

    The upstream builder requires FlashMLA during config construction, while
    Ascend uses dlinfer's Lightning Indexer instead.  Temporarily report
    FlashMLA as available while the upstream builder runs, then restore the
    non-FlashMLA model semantics for the Ascend runtime.
    """
    from lmdeploy.pytorch.configurations import deepseek_v2 as deepseek_v2_config
    from lmdeploy.pytorch.configurations.deepseek_v32 import DeepseekV32ModelConfigBuilder

    if getattr(DeepseekV32ModelConfigBuilder, '_dlinfer_ascend_patched', False):
        return

    original_build = DeepseekV32ModelConfigBuilder.build
    original_flash_mla_available = deepseek_v2_config.flash_mla_available

    @classmethod
    def custom_build(cls, hf_config, model_path: str | None = None, **kwargs):
        device_type = kwargs.get('device_type', 'auto')
        if device_type not in ('ascend', 'npu'):
            return original_build(hf_config, model_path=model_path, **kwargs)

        deepseek_v2_config.flash_mla_available = lambda: True
        try:
            config = original_build(hf_config, model_path=model_path, **kwargs)
        finally:
            deepseek_v2_config.flash_mla_available = original_flash_mla_available
            hf_config.use_flash_mla = False

        # Ascend uses dlinfer attention/indexer, not the CUDA FlashMLA path.
        config.use_flash_mla = False
        return config

    DeepseekV32ModelConfigBuilder.build = custom_build
    DeepseekV32ModelConfigBuilder._dlinfer_ascend_patched = True


def patch_glm_moe_dsa_config():
    """Load Ascend ModelSlim metadata in the dlinfer configuration patch."""
    import json
    import os

    from lmdeploy.pytorch.configurations.glm_moe_dsa import GlmMoeDsaModelConfigBuilder
    from lmdeploy.utils import get_logger

    logger = get_logger('lmdeploy')

    if getattr(GlmMoeDsaModelConfigBuilder, '_dlinfer_modelslim_patched', False):
        return

    original_build = GlmMoeDsaModelConfigBuilder.build

    @classmethod
    def custom_build(cls, hf_config, model_path: str | None = None, **kwargs):
        device_type = kwargs.get('device_type', 'auto')
        modelslim_path = (os.path.join(model_path, 'quant_model_description.json')
                          if model_path else None)
        if (device_type in ('ascend', 'npu') and modelslim_path
                and os.path.isfile(modelslim_path)):
            with open(modelslim_path, encoding='utf-8') as f:
                quant_description = json.load(f)
            if not isinstance(quant_description, dict):
                raise TypeError(f'Expected a JSON object in {modelslim_path}.')
            hf_config.quantization_config = {
                'quant_method': 'modelslim',
                'quant_dtype': 'int8',
                'quant_description': quant_description,
            }
            logger.info(f'Using Ascend ModelSlim quantization metadata from {modelslim_path}.')
        return original_build(hf_config, model_path=model_path, **kwargs)

    GlmMoeDsaModelConfigBuilder.build = custom_build
    GlmMoeDsaModelConfigBuilder._dlinfer_modelslim_patched = True


def patch_deepseek_v32_qkv():
    """Use separate Q-A and KV-A projections on Ascend.

    CUDA uses the merged ``fused_qkv_a_proj`` operator.  ModelSlim W8A8
    checkpoints contain independent quantization metadata for ``q_a_proj``
    and ``kv_a_proj_with_mqa``, while Ascend does not provide the merged
    operator.  Keep this projection choice local to dlinfer.
    """
    from lmdeploy.pytorch.models import deepseek_v32

    attention_cls = deepseek_v32.DeepseekV32Attention
    if getattr(attention_cls, '_dlinfer_ascend_qkv_patched', False):
        return

    original_init = attention_cls.__init__

    def custom_init(self,
                    config,
                    layer_idx,
                    dtype=None,
                    device=None,
                    all_reduce=True,
                    prefix=''):
        if config.q_lora_rank is None:
            return original_init(self,
                                 config,
                                 layer_idx,
                                 dtype=dtype,
                                 device=device,
                                 all_reduce=all_reduce,
                                 prefix=prefix)

        deepseek_v32.nn.Module.__init__(self)
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        num_key_value_heads = getattr(config, 'num_key_value_heads', 1)
        use_flash_mla = getattr(config, 'use_flash_mla', False)

        self.q_a_proj = deepseek_v32.build_colwise_linear(
            self.hidden_size,
            config.q_lora_rank,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            prefix=f'{prefix}.q_a_proj' if prefix else '',
        )
        self.q_a_layernorm = deepseek_v32.RMSNorm(
            config.q_lora_rank,
            1e-6,
            quant_config=quantization_config,
            dtype=deepseek_v32.torch.float32,
            device=device,
        )
        self.q_b_proj = deepseek_v32.build_colwise_linear(
            config.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            prefix=f'{prefix}.q_b_proj' if prefix else '',
        )
        self.kv_a_proj_with_mqa = deepseek_v32.build_colwise_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            prefix=f'{prefix}.kv_a_proj_with_mqa' if prefix else '',
        )
        self.kv_a_layernorm = deepseek_v32.RMSNorm(
            config.kv_lora_rank,
            1e-6,
            quant_config=quantization_config,
            dtype=deepseek_v32.torch.float32,
            device=device,
        )
        self.kv_b_proj = deepseek_v32.build_colwise_linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            prefix=f'{prefix}.kv_b_proj' if prefix else '',
        )
        self.kc = deepseek_v32.DeepseekV2BMM(
            self.num_heads,
            config.qk_nope_head_dim,
            config.kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        self.apply_rotary_pos_emb = deepseek_v32.ApplyRotaryEmb()
        self.softmax_scale = self.q_head_dim**-0.5

        rope_scaling = deepseek_v32.get_rope_parameters(config)
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get('mscale_all_dim', 0)
            if mscale_all_dim:
                scaling_factor = rope_scaling['factor']
                mscale = deepseek_v32.yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_fwd = deepseek_v32.Attention(
            self.num_heads,
            config.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.softmax_scale,
            num_kv_heads=num_key_value_heads,
            v_head_size=config.kv_lora_rank,
            num_replicate_kv_heads=num_replicate_kv_heads,
            use_flash_mla=use_flash_mla,
            mla_index_topk=config.index_topk,
        )
        self.vc = deepseek_v32.DeepseekV2BMM(
            self.num_heads,
            config.kv_lora_rank,
            self.v_head_dim,
            dtype=dtype,
            device=device,
        )
        self.o_proj = deepseek_v32.build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            all_reduce=all_reduce,
            prefix=f'{prefix}.o_proj' if prefix else '',
        )
        self.indexer = self._build_indexer(config, layer_idx, dtype, device, prefix)

    def custom_qkv_proj(self, hidden_states, num_heads):
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        if self.q_lora_rank is None:
            q_a_states = hidden_states
            key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        else:
            q_a_states = self.q_a_proj(hidden_states)
            key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])

        query_states, q_pe, qr = self._q_proj(q_a_states, num_heads, nope_size, pe_size)
        key_states, value_states, k_pe = self._kv_proj(key_states, nope_size)
        return query_states, key_states, value_states, q_pe, k_pe, qr

    attention_cls.__init__ = custom_init
    attention_cls._qkv_proj = custom_qkv_proj
    attention_cls._dlinfer_ascend_qkv_patched = True


def patch_glm_moe_dsa_norm_dtype():
    """Cast GLM normalization layers to the model dtype on Ascend."""
    from lmdeploy.pytorch.models.glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaModel,
    )

    if getattr(GlmMoeDsaModel, '_dlinfer_norm_dtype_patched', False):
        return

    original_decoder_init = GlmMoeDsaDecoderLayer.__init__
    original_model_init = GlmMoeDsaModel.__init__

    def custom_decoder_init(self,
                            config,
                            layer_idx,
                            dtype=None,
                            device=None,
                            prefix=''):
        original_decoder_init(self,
                              config,
                              layer_idx,
                              dtype=dtype,
                              device=device,
                              prefix=prefix)
        if dtype is not None:
            self.input_layernorm.to(dtype=dtype)
            self.post_attention_layernorm.to(dtype=dtype)

    def custom_model_init(self, config, dtype=None, device=None):
        original_model_init(self, config, dtype=dtype, device=device)
        if dtype is not None:
            self.norm.to(dtype=dtype)

    GlmMoeDsaDecoderLayer.__init__ = custom_decoder_init
    GlmMoeDsaModel.__init__ = custom_model_init
    GlmMoeDsaModel._dlinfer_norm_dtype_patched = True


def patch_deepseek_v2_moe():
    """Use the Ascend EP reduction semantics for DeepSeek MoE."""
    from lmdeploy.pytorch.models import deepseek_v2

    moe_cls = deepseek_v2.DeepseekV2MoE
    if getattr(moe_cls, '_dlinfer_ascend_moe_patched', False):
        return

    original_init = moe_cls.__init__

    def custom_init(self,
                    config,
                    layer_idx,
                    dtype=None,
                    device=None,
                    all_reduce=True,
                    prefix=''):
        original_init(self,
                      config,
                      layer_idx,
                      dtype=dtype,
                      device=device,
                      all_reduce=all_reduce,
                      prefix=prefix)

        dist_ctx = deepseek_v2.get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        self._all_reduce = (all_reduce and dist_config.dp == 1
                            and dist_config.world_size > 1
                            and dist_config.ep == 1)
        self._all_reduce_shared_experts = (
            all_reduce and dist_config.dp == 1 and dist_config.ep > 1
            and dist_config.mlp_tp > 1)
        self._shared_expert_tp_group = None
        if self._all_reduce_shared_experts:
            self._shared_expert_tp_group = dist_ctx.mlp_tp_group.gpu_group

    def custom_forward(self, hidden_states, all_routed_experts=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        routed_experts = None
        if all_routed_experts is not None:
            routed_experts = all_routed_experts[:, self.layer_idx, :]
        topk_weights, topk_ids = self.gate(
            hidden_states, routed_experts=routed_experts)

        out_states = self.experts(hidden_states, topk_weights, topk_ids)
        if self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
            # EP already combines routed expert outputs. Only the shared expert
            # output remains sharded over the MLP TP group.
            if self._all_reduce_shared_experts:
                deepseek_v2.dist.all_reduce(
                    shared_states, group=self._shared_expert_tp_group)
            out_states += shared_states
        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self._all_reduce:
            deepseek_v2.dist.all_reduce(out_states)
        return out_states

    moe_cls.__init__ = custom_init
    moe_cls.forward = custom_forward
    moe_cls._dlinfer_ascend_moe_patched = True


def patch_deepseek_v2_modelslim_weight_loader():
    """Adapt ModelSlim auxiliary checkpoint tensors on Ascend."""
    from lmdeploy.pytorch.models import deepseek_v2

    model_cls = deepseek_v2.DeepseekV2ForCausalLM
    if getattr(model_cls, '_dlinfer_modelslim_weight_loader_patched', False):
        return

    original_load_weight_attention = model_cls._load_weight_attention
    original_load_weights = model_cls.load_weights

    def map_modelslim_param_name(self, name, params_dict):
        quantization_config = getattr(self.config,
                                      'quantization_config', None) or {}
        if quantization_config.get('quant_method') != 'modelslim':
            return name
        if name.endswith('.weight_offset'):
            return None
        if name.endswith('.weight_scale'):
            mapped_name = name.removesuffix('.weight_scale') + '.scale'
            return mapped_name if mapped_name in params_dict else None
        return name

    def custom_load_weight_experts(self, name, loaded_weight, params_dict,
                                   expert_params_mapping):
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            name = self._map_modelslim_param_name(name, params_dict)
            if name is None:
                return
            param = params_dict[name]
            deepseek_v2.load_weight(
                param,
                loaded_weight,
                expert_id=expert_id,
                shard_id=shard_id,
            )
            break
        else:
            name = self._map_modelslim_param_name(name, params_dict)
            if name is None:
                return
            deepseek_v2.load_weight(params_dict[name], loaded_weight)

    def custom_load_weight_attention(self, name, loaded_weight, params_dict,
                                     update_pe_mapping):
        mapped_name = self._map_modelslim_param_name(name, params_dict)
        if mapped_name is None:
            return
        # Input quantization metadata is shared by the whole projection and
        # has shape [1].  It must not enter DeepSeek's output-channel RoPE
        # permutation, which expects dim 0 to be divisible by head_dim.
        if mapped_name.endswith(('.input_scale', '.input_offset')):
            deepseek_v2.load_weight(params_dict[mapped_name], loaded_weight)
            return
        # Delegate output-channel metadata (for example a mapped dynamic
        # weight scale) using its actual parameter name so that it receives
        # the same RoPE permutation as the corresponding projection weight.
        return original_load_weight_attention(
            self,
            mapped_name,
            loaded_weight,
            params_dict,
            update_pe_mapping,
        )

    def custom_load_weights(self, weights):
        quantization_config = getattr(self.config,
                                      'quantization_config', None) or {}
        if quantization_config.get('quant_method') != 'modelslim':
            return original_load_weights(self, weights)

        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            ('.gate_up_proj', '.gate_proj'),
            ('.gate_up_proj', '.up_proj'),
        ]
        if not getattr(self.config, 'use_mla', True):
            stacked_params_mapping.extend([
                ('.qkv_proj', '.q_proj'),
                ('.qkv_proj', '.k_proj'),
                ('.qkv_proj', '.v_proj'),
            ])

        def convert_weights():
            for name, loaded_weight in weights:
                is_attention = ('.self_attn' in name
                                and getattr(self.config, 'use_mla', True))
                if '.experts' in name or is_attention:
                    yield name, loaded_weight
                    continue
                if name.endswith('.weight_offset'):
                    continue
                if name.endswith('.weight_scale'):
                    mapped_name = name.removesuffix('.weight_scale') + '.scale'
                    param_name = mapped_name
                    for fused_name, shard_name in stacked_params_mapping:
                        if shard_name in param_name:
                            param_name = param_name.replace(shard_name,
                                                            fused_name)
                            break
                    if param_name not in params_dict:
                        continue
                    name = mapped_name
                yield name, loaded_weight

        return original_load_weights(self, convert_weights())

    model_cls._map_modelslim_param_name = map_modelslim_param_name
    model_cls._load_weight_experts = custom_load_weight_experts
    model_cls._load_weight_attention = custom_load_weight_attention
    model_cls.load_weights = custom_load_weights
    model_cls._dlinfer_modelslim_weight_loader_patched = True


def patch_glm_moe_dsa_weight_loader():
    """Ignore the ModelSlim QuaRot-only MTP weight on Ascend."""
    from lmdeploy.pytorch.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    if getattr(GlmMoeDsaForCausalLM,
               '_dlinfer_ascend_weight_loader_patched', False):
        return

    original_load_weights = GlmMoeDsaForCausalLM.load_weights

    def custom_load_weights(self, weights):
        weights = ((name, weight) for name, weight in weights
                   if name != 'rot.weight')
        return original_load_weights(self, weights)

    GlmMoeDsaForCausalLM.load_weights = custom_load_weights
    GlmMoeDsaForCausalLM._dlinfer_ascend_weight_loader_patched = True


def patch_glm_moe_dsa_indexer():
    """Use the Ascend unfused, non-Hadamard DSA indexer path.

    The common GLM implementation keeps CUDA's fused projection and
    Hadamard preprocessing semantics.  Ascend's Lightning Indexer consumes
    the separate BF16 projection outputs directly, so this adaptation stays
    local to the dlinfer device patch.
    """
    from lmdeploy.pytorch import envs as lmdeploy_envs
    from lmdeploy.pytorch.models.glm_moe_dsa import GlmMoeDsaIndexer

    if getattr(GlmMoeDsaIndexer, '_dlinfer_ascend_patched', False):
        return

    original_init = GlmMoeDsaIndexer.__init__

    def custom_init(self,
                    config,
                    layer_idx,
                    dtype=None,
                    device=None,
                    prefix=''):
        # Force the common constructor to materialize wk and weights_proj.
        original_disable = lmdeploy_envs.disable_dsa_indexer_fusion
        lmdeploy_envs.disable_dsa_indexer_fusion = True
        try:
            original_init(self,
                          config,
                          layer_idx,
                          dtype=dtype,
                          device=device,
                          prefix=prefix)
        finally:
            lmdeploy_envs.disable_dsa_indexer_fusion = original_disable
        self.use_fusion = False

    def custom_forward(self, x, qr, freqs_cis, attn_metadata=None):
        # This is the common unfused path without CUDA-only Hadamard rotation.
        q = self.wq_b(qr).unflatten(-1, (-1, self.head_dim))
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe, k_pe = self._apply_rotary_pos_emb(q_pe, k_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe[0], k_nope[0, :, None]], dim=-1)
        weights = self.weights_proj(x) * self.n_heads**-0.5
        return self.indexer_topk(q[0],
                                 k[:, 0],
                                 weights[0],
                                 attn_metadata=attn_metadata)

    GlmMoeDsaIndexer.__init__ = custom_init
    GlmMoeDsaIndexer.forward = custom_forward
    GlmMoeDsaIndexer._dlinfer_ascend_patched = True


##### patch cache engine #####


def patch_glm_moe_dsa_split_cache():
    """Use independent contiguous noPE and RoPE caches for Ascend DSA."""
    from lmdeploy.pytorch.configurations.glm_moe_dsa import (
        GlmMoeDsaModelConfigBuilder,
    )
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.models.glm_moe_dsa import GlmMoeDsaAttention

    if getattr(GlmMoeDsaModelConfigBuilder,
               '_dlinfer_split_cache_patched', False):
        return

    original_build = GlmMoeDsaModelConfigBuilder.build

    @classmethod
    def custom_build(cls,
                     hf_config,
                     model_path: str | None = None,
                     **kwargs):
        config = original_build(hf_config,
                                model_path=model_path,
                                **kwargs)
        # Cache only the RoPE key in K and the latent/noPE value in V.  Their
        # combined width is unchanged, but each cache can now be contiguous.
        config.k_head_dim = hf_config.qk_rope_head_dim
        config.v_head_dim = hf_config.kv_lora_rank
        config.split_mla_kv_cache = True
        return config

    def custom_forward(
        self,
        hidden_states,
        rotary_pos_emb,
        past_key_value=None,
        attn_metadata=None,
        topk_indices_buffer=None,
        skip_topk: bool = False,
    ):
        dist_config = get_dist_manager().current_config()
        num_heads = (self.num_heads if dist_config.dp > 1 else
                     self.num_heads // dist_config.attn_tp)
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        query_states, key_states, value_states, q_pe, k_pe, qr = (
            self._qkv_proj(hidden_states, num_heads=num_heads))
        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe,
                                               k_pe,
                                               cos,
                                               sin,
                                               inplace=False)
        query_states[..., nope_size:] = q_pe
        key_states[..., nope_size:] = k_pe

        if topk_indices_buffer is None:
            raise RuntimeError(
                f'Layer {self.layer_idx} requires a DSA top-k indices buffer.')
        if self.indexer is not None and not skip_topk:
            topk_indices = topk_indices_buffer.write(
                self.indexer(hidden_states,
                             qr,
                             rotary_pos_emb,
                             attn_metadata=attn_metadata))
        else:
            topk_indices = topk_indices_buffer.read(q_len,
                                                    hidden_states.device)

        rope_cache, nope_cache = past_key_value[:2]

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            rope_cache,
            nope_cache,
            attn_metadata,
            k_scales_zeros=(None if len(past_key_value) == 2 else
                            past_key_value[2]),
            v_scales_zeros=(None if len(past_key_value) == 2 else
                            past_key_value[3]),
            nsa_indices=topk_indices,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads,
                                             self.v_head_dim)
        self.vc(attn_output, attn_bmm_out)
        return self.o_proj(attn_bmm_out.flatten(-2, -1)[None])

    GlmMoeDsaModelConfigBuilder.build = custom_build
    GlmMoeDsaModelConfigBuilder._dlinfer_split_cache_patched = True
    GlmMoeDsaAttention.forward = custom_forward


def patch_gated_delta_net():
    import torch
    import torch.nn.functional as F
    from typing import Any, Sequence, Tuple
    from torch.profiler import record_function

    from lmdeploy.pytorch.nn import gated_delta
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta
    from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

    class AscendGatedDeltaMeta:

        def __init__(
            self,
            num_tokens: int,
            conv_kernel_size: int,
            state_ids: torch.Tensor,
            attn_metadata: Any,
        ):
            self.is_decoding = attn_metadata.is_decoding
            self.cu_seqlens = attn_metadata.cu_seqlens_q
            self.is_multi_token_decoding = attn_metadata.is_multi_token_decoding
            self.max_q_seq_len = attn_metadata.max_q_seqlen

            self.num_spec_tokens = get_step_ctx_manager().build_ctx.num_spec_tokens
            self.cache_seqlens = getattr(attn_metadata, "cache_seqlens", None)
            self.spec_state_offsets = getattr(attn_metadata, "spec_state_offsets", None)
            self.spec_conv_offsets = getattr(attn_metadata, "spec_conv_offsets", None)

            self.state_ids = state_ids.clamp(0)
            self.has_initial_state = attn_metadata.has_initial_state
            self.conv_state_indices = self.state_ids.to(torch.int32)

    def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
        try:
            from dlinfer.vendor.ascend.triton_ops import RMSNormGated
        except Exception:
            raise RuntimeError(
                "Triton is not installed or Ascend triton_ops failed to load. "
                "Please install triton-ascend to use this feature."
            )

        device = kwargs["device"]
        return RMSNormGated(hidden_size, eps=eps, norm_before_gate=True, device=device)

    class AscendCausalConv1dFunc:

        def __init__(self, activation: str = "silu"):
            try:
                from dlinfer.vendor.ascend.triton_ops import (
                    causal_conv1d_fn,
                    causal_conv1d_update_npu,
                )

                self.causal_conv1d_fn = causal_conv1d_fn
                self.causal_conv1d_update = causal_conv1d_update_npu
            except Exception:
                raise RuntimeError(
                    "Triton is not installed or Ascend triton_ops failed to load. "
                    "Please install triton-ascend to use this feature."
                )

            self.activation = activation

        def conv1d_func(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """
            x: (b, seqlen, dim)
            seqlen: (b)
            out: (b, seqlen, dim)
            conv_state: (b, dim, kernel_size)
            """
            spec_conv_offsets = getattr(gated_delta_meta, "spec_conv_offsets", None)
            if spec_conv_offsets is not None:
                read_conv_offsets, write_conv_offsets = spec_conv_offsets
            else:
                read_conv_offsets, write_conv_offsets = None, None

            out = self.causal_conv1d_fn(
                x.t(),
                weight,
                bias,
                activation=self.activation,
                conv_states=conv_state.transpose(1, 2),
                has_initial_state=gated_delta_meta.has_initial_state,
                cache_indices=gated_delta_meta.conv_state_indices,
                query_start_loc=gated_delta_meta.cu_seqlens,
                read_conv_offsets=read_conv_offsets,
                write_conv_offsets=write_conv_offsets,
            )

            out = out.t().unsqueeze(0)

            return out, conv_state

        # 替换
        def conv1d_update(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            conv_state_indices: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            update_kwargs = {}
            validate_data = True

            cache_seqlens = gated_delta_meta.cache_seqlens
            is_multi_token_decoding = gated_delta_meta.is_multi_token_decoding

            if is_multi_token_decoding:
                # Ring-buffer decode path: positions are derived from cache_seqlens.
                update_kwargs["cache_seqlens"] = gated_delta_meta.cache_seqlens
                # Multi-token decode uses varlen format (2-D x tensor); must keep
                # IS_VARLEN=True by passing query_start_loc, otherwise x gets incorrectly
                # unsqueezed and cache_seqlens is accessed out-of-bounds.
                update_kwargs["query_start_loc"] = gated_delta_meta.cu_seqlens
                update_kwargs["max_query_len"] = gated_delta_meta.max_q_seq_len
                validate_data = False

            out = self.causal_conv1d_update(
                x,
                conv_state,
                weight.t().contiguous(),
                bias,
                self.activation,
                conv_state_indices=conv_state_indices,
                validate_data=validate_data,
                **update_kwargs,
            )
            return out.unsqueeze(0), conv_state

        @record_function("causal_conv1d")
        def __call__(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            weight_reshaped = weight.squeeze(1)
            x = x.squeeze(0)

            if gated_delta_meta.is_decoding or gated_delta_meta.is_multi_token_decoding:
                conv_state_indices = gated_delta_meta.conv_state_indices
                return self.conv1d_update(
                    x,
                    weight_reshaped,
                    bias,
                    conv_state,
                    conv_state_indices,
                    gated_delta_meta,
                )
            return self.conv1d_func(
                x, weight_reshaped, bias, conv_state, gated_delta_meta=gated_delta_meta
            )

    class AscendGatedDelta:

        def __init__(self, use_qk_l2norm_in_kernel: bool = True):
            try:
                from dlinfer.vendor.ascend.triton_ops import (
                    chunk_gated_delta_rule,
                    fused_sigmoid_gating_delta_rule_update,
                    fused_recurrent_gated_delta_rule,
                )

                self.chunk_gated_delta_rule = chunk_gated_delta_rule
                self.fused_sigmoid_gating_delta_rule_update = (
                    fused_sigmoid_gating_delta_rule_update
                )
                self.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
            except Exception:
                raise RuntimeError(
                    "Triton is not installed or Ascend triton_ops failed to load. "
                    "Please install triton-ascend and triton-ascend-kernels to use this feature."
                )

            self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            recurrent_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """call."""

            is_decoding = gated_delta_meta.is_decoding
            is_multi_token_decoding = gated_delta_meta.is_multi_token_decoding

            if is_decoding:
                core_attn_out = self.fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a.contiguous(),
                    b=b.contiguous(),
                    initial_state_source=recurrent_state,
                    initial_state_indices=gated_delta_meta.state_ids,
                    cu_seqlens=gated_delta_meta.cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
                return core_attn_out, None
            elif is_multi_token_decoding:

                beta = b.sigmoid()
                # If the model is loaded in fp16, without the .float() here, A might be -inf
                g = (-A_log.float().exp()) * F.softplus(a.float() + dt_bias)

                state_slots = recurrent_state.size(1)
                flat_recurrent_state = recurrent_state.view(
                    -1, *recurrent_state.shape[2:]
                )
                core_attn_out, _ = self.fused_recurrent_gated_delta_rule(
                    q=query.contiguous(),
                    k=key.contiguous(),
                    v=value.contiguous(),
                    g=g.contiguous(),
                    beta=beta.contiguous(),
                    initial_state=flat_recurrent_state,
                    inplace_final_state=True,
                    cu_seqlens=gated_delta_meta.cu_seqlens,
                    cache_seqlens_rb=gated_delta_meta.cache_seqlens,
                    state_ids_rb=gated_delta_meta.state_ids,
                    num_state=state_slots,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
                return core_attn_out, None
            else:

                beta = b.sigmoid()
                # If the model is loaded in fp16, without the .float() here, A might be -inf
                g = (-A_log.float().exp()) * F.softplus(a.float() + dt_bias)

                if gated_delta_meta.spec_state_offsets is not None:
                    state_ids = gated_delta_meta.state_ids
                    # Circular-buffer read slot: history_len % NUM_STATE
                    read_slots = gated_delta_meta.spec_state_offsets[0]
                    initial_state = (
                        recurrent_state[state_ids, read_slots]
                        .transpose(-1, -2)
                        .contiguous()
                    )
                else:
                    initial_state = recurrent_state[gated_delta_meta.state_ids]
                initial_state[~gated_delta_meta.has_initial_state, ...] = 0
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=gated_delta_meta.cu_seqlens,
                    head_first=False,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
                if gated_delta_meta.spec_state_offsets is not None:
                    state_ids = gated_delta_meta.state_ids
                    # Circular-buffer write slot: (history_len + query_len) % NUM_STATE
                    write_slots = gated_delta_meta.spec_state_offsets[1]
                    recurrent_state[state_ids, write_slots] = (
                        last_recurrent_state.transpose(-1, -2).to(recurrent_state.dtype)
                    )
                else:
                    recurrent_state[gated_delta_meta.state_ids] = (
                        last_recurrent_state.to(recurrent_state.dtype)
                    )
                return core_attn_out, last_recurrent_state

    gated_delta.GatedDeltaMeta = AscendGatedDeltaMeta
    gated_delta.CausalConv1dFunc = AscendCausalConv1dFunc
    gated_delta.GatedDelta = AscendGatedDelta
    gated_delta.build_rmsnorm_gated = build_rmsnorm_gated


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def patch_qwen3_5():
    import torch
    from typing import List

    from lmdeploy.utils import is_bf16_supported
    from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder
    from lmdeploy.pytorch.configurations.qwen3_next import _check_env_qwen3_next
    from lmdeploy.vl.constants import Modality

    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta, CausalConv1d
    from lmdeploy.pytorch.model_inputs import StepContext
    from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder
    from lmdeploy.pytorch.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5GatedDeltaNet,
    )

    @classmethod
    def custom_build(
        cls,
        hf_config,
        model_path: str = None,
        tp: int = 1,
        is_draft_model: bool = False,
        spec_method: str = None,
        num_spec_tokens: int = 0,
        **kwargs,
    ):
        """build."""
        text_config = hf_config.text_config
        # propagate quantization_config from top-level hf_config into text_config
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config is not None and not hasattr(
            text_config, "quantization_config"
        ):
            text_config.quantization_config = quantization_config
        cfg = DefaultModelConfigBuilder.build(text_config, model_path, tp=tp, **kwargs)

        if getattr(hf_config.text_config, "attn_output_gate", False):
            cfg.num_attention_heads *= 2
        # update num layers
        num_layers = cfg.num_layers
        layer_types = text_config.layer_types
        num_delta_layers = sum([1 for lt in layer_types if lt == "linear_attention"])
        num_full_layers = num_layers - num_delta_layers
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = text_config.linear_key_head_dim
        head_v_dim = text_config.linear_value_head_dim
        num_v_heads = text_config.linear_num_value_heads // tp
        num_k_heads = text_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim + num_spec_tokens

        # Ascend Patch
        conv_state_shape = (conv_kernel_size, conv_dim)
        if num_spec_tokens > 0:
            recurrent_state_shape = (
                1 + num_spec_tokens,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
        else:
            recurrent_state_shape = (num_v_heads, head_k_dim, head_v_dim)

        device_type = kwargs.get("device_type", "auto")
        if is_bf16_supported(device_type):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Ascend Patch
        # Use per-layer shapes so each cache slice is (num_caches, ...) — contiguous by
        # construction. Storing num_delta_layers as the first shape dim would require a
        # transpose later, producing non-contiguous views.
        cfg.states_shapes = [(conv_state_shape, dtype)] * num_delta_layers + [
            (recurrent_state_shape, torch.float32)
        ] * num_delta_layers

        cfg.is_gated_delta = True
        cfg.check_env_func = _check_env_qwen3_next

        cfg.use_mrope = True

        # Speculative decoding support
        if spec_method is not None:
            assert spec_method == "qwen3_5_mtp"
            cfg.model_paradigm = "ar_spec"

        # draft model cfg
        if is_draft_model:
            hf_config.architectures[0] = "Qwen3_5MTPModel"
            # remove for correct mapping when building the patched model
            if hasattr(hf_config, "auto_map"):
                del hf_config.auto_map

            cfg.model_paradigm = "ar_spec"
            cfg.num_layers = text_config.mtp_num_hidden_layers
            cfg.states_shapes = []

        return cfg

    def custom_prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # Ascend Patch
        # make past_key_values
        # state_caches holds num_delta_layers conv entries then num_delta_layers recurrent
        # entries, each already shaped (num_caches, ...) and contiguous.
        n = len(context.state_caches) // 2
        state_caches = list(zip(context.state_caches[:n], context.state_caches[n:]))

        past_key_values = list(past_key_values)
        new_past_key_values = []
        for layer_type in self.config.text_config.layer_types:
            if layer_type == "linear_attention":
                new_past_key_values.append(state_caches.pop(0))
            elif layer_type == "full_attention":
                new_past_key_values.append(past_key_values.pop(0))

        # vlm inputs
        pixel_values = None
        vis_cu_seqlens = None
        vis_pos_emb = None
        multimodal_mask = None
        grid_thw = None
        pos_embeds = None
        # for time series
        ts_values = None
        ts_lens = None
        ts_sr = None
        if context.input_multimodals is not None:
            mm_inputs = [
                input_mm.get("mm_data", []) for input_mm in context.input_multimodals
            ]
            # flatten batch
            mm_inputs = [item for sublist in mm_inputs for item in sublist]

            if len(mm_inputs) > 0:
                modality = mm_inputs[0].modality
                multimodal_mask = self.get_multimodal_mask(input_ids, mm_inputs)

                if modality == Modality.TIME_SERIES:
                    ts_values = torch.cat([inp.data for inp in mm_inputs])
                    ts_lens = torch.cat([inp.meta["ts_lens"] for inp in mm_inputs])
                    ts_sr = torch.cat([inp.meta["ts_sr"] for inp in mm_inputs])
                else:
                    pixel_values = torch.cat([inp.data for inp in mm_inputs])
                    grid_thw = torch.stack(
                        [data.meta["grid_thw"] for data in mm_inputs]
                    ).cpu()
                    vis_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
                    pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
                    vis_cu_seqlens = torch.repeat_interleave(
                        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                    ).to(pixel_values.device)
                    vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                    vis_pos_emb = vis_pos_emb.repeat(1, 2)
                    vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        mrope_position_ids = getattr(context, "mrope_position_ids", None)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(
                inputs_embeds
            )

        # return input embeds for spec decoding
        return_input_embeds = self.is_spec_decoding and (
            pixel_values is not None or context.is_chunk_multimodal
        )

        # return input embeds for spec decoding
        return_input_embeds = self.is_spec_decoding and (
            pixel_values is not None or context.is_chunk_multimodal
        )

        # return input embeds for spec decoding
        return_input_embeds = self.is_spec_decoding and (
            pixel_values is not None or context.is_chunk_multimodal
        )

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            # vl inputs
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            multimodal_mask=multimodal_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
            return_input_embeds=return_input_embeds,
            # for time series
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
        )

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        gated_delta_meta: GatedDeltaMeta,
    ):
        """forward."""

        # load states
        conv_state, recurrent_state = self._load_state(past_key_value, gated_delta_meta)

        # inputs proj
        projected_states_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        # [..., ng, np/ng * hn] -> [..., np, hn]
        z = z.unflatten(-1, (-1, self.head_v_dim))
        projected_states_ba = self.in_proj_ba(hidden_states)
        b, a = self.fix_ba_ordering(projected_states_ba)

        mixed_qkv = projected_states_qkv
        mixed_qkv, conv_state = self.conv1d(
            mixed_qkv, conv_state, gated_delta_meta=gated_delta_meta
        )

        tp = (self.key_dim * 2 + self.value_dim) // mixed_qkv.size(-1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // tp,
                self.key_dim // tp,
                self.value_dim // tp,
            ],
            dim=-1,
        )
        query = query.unflatten(-1, (-1, self.head_k_dim))
        key = key.unflatten(-1, (-1, self.head_k_dim))
        value = value.unflatten(-1, (-1, self.head_v_dim))

        # beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # g = self.get_A_log_exp() * F.softplus(a.float() + self.dt_bias)
        # if self.kv_ratio > 1:
        # query = query.repeat_interleave(self.kv_ratio, dim=-2)
        # key = key.repeat_interleave(self.kv_ratio, dim=-2)

        core_attn_out, recurrent_state = self.gated_delta(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.A_log,
            self.dt_bias,
            a,
            b,
            recurrent_state=recurrent_state,
            gated_delta_meta=gated_delta_meta,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    Qwen3_5GatedDeltaNet.forward = custom_forward
    Qwen3_5ModelConfigBuilder.build = custom_build
    Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation = (
        custom_prepare_inputs_for_generation
    )


def vendor_device_init():
    import_vendor_module(vendor_name)
    patch_compiled_func()
    patch_async_sampling_logits()
    if vendor_name == "ascend":
        patch_rejection_sampler()
        patch_modelslim_quantization_config()
        patch_glm_moe_dsa_config()
        patch_deepseek_v32_config()
        patch_deepseek_v32_qkv()
        patch_glm_moe_dsa_norm_dtype()
        patch_deepseek_v2_moe()
        patch_deepseek_v2_modelslim_weight_loader()
        patch_glm_moe_dsa_weight_loader()
        patch_glm_moe_dsa_indexer()
        patch_glm_moe_dsa_split_cache()
        patch_gated_delta_net()
        patch_qwen3_5()


vendor_device_init()
