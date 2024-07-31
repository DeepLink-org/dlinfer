import transformers

def apply_model_patches(module, name):
    if name == 'transformers_modules.internlm2-chat-7b.modeling_internlm2':
        from . import internlm2
        module.InternLM2RMSNorm.forward = internlm2.modeling_internlm2_InternLM2RMSNorm_forward
        module.InternLM2Attention.forward = internlm2.modeling_internlm2_InternLM2Attention_forward
        module.InternLM2ForCausalLM.prepare_inputs_for_generation = internlm2.modeling_internlm2_InternLM2ForCausalLM_prepare_inputs_for_generation
        transformers.cache_utils.DynamicCache.update = internlm2.transformers_cache_utils_dynamiccache_update
 

