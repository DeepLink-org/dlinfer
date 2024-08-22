import transformers
import inspect

def apply_model_patches(module):
    if module.__name__ == 'transformers_modules.internlm2-chat-7b.modeling_internlm2':
        from . import internlm2
        module.InternLM2RMSNorm.forward = internlm2.modeling_internlm2_InternLM2RMSNorm_forward
        module.InternLM2Attention.forward = internlm2.modeling_internlm2_InternLM2Attention_forward
        module.InternLM2ForCausalLM.prepare_inputs_for_generation = internlm2.modeling_internlm2_InternLM2ForCausalLM_prepare_inputs_for_generation
        transformers.cache_utils.DynamicCache.update = internlm2.transformers_cache_utils_dynamiccache_update
    elif module.__name__ == 'transformers_modules.modeling_internvl_chat':
        from . import internvl
        module = inspect.getmodule(module.InternVisionModel)
        module.InternAttention._naive_attn = internvl.InternAttention_naive_attn
        module.InternRMSNorm.forward = internvl.InternRMSNorm_forward
    elif module.__name__ == 'transformers_modules.cogvlm-chat.modeling_cogvlm':
        from . import cogvlm_ascend
        module.EVA2CLIPModel = cogvlm_ascend.PatchedEVA2CLIPModel

