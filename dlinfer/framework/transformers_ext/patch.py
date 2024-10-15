# Copyright (c) 2024, DeepLink. All rights reserved.
import transformers
import inspect


def apply_model_patches(module):
    if module.__name__.endswith(".modeling_internlm2"):
        from . import internlm2

        module.InternLM2RMSNorm.forward = (
            internlm2.modeling_internlm2_InternLM2RMSNorm_forward
        )
        module.InternLM2Attention.forward = (
            internlm2.modeling_internlm2_InternLM2Attention_forward
        )
        module.InternLM2ForCausalLM.prepare_inputs_for_generation = (
            internlm2.modeling_internlm2_InternLM2ForCausalLM_prepare_inputs_for_generation
        )
        transformers.cache_utils.DynamicCache.update = (
            internlm2.transformers_cache_utils_dynamiccache_update
        )
    elif module.__name__.endswith(".modeling_internvl_chat"):
        from . import internvl

        vit_module = inspect.getmodule(module.InternVisionModel)
        vit_module.InternAttention._naive_attn = internvl.InternAttention_naive_attn
        vit_module.InternRMSNorm.forward = internvl.InternRMSNorm_forward
    elif module.__name__.endswith(".modeling_cogvlm"):
        from . import cogvlm

        # get parent module from another source code file
        vit_module = inspect.getmodule(module.EVA2CLIPModel)
        vit_module.Attention.forward = cogvlm.PatchedAttention_forward
