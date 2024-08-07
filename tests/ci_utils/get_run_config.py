from lmdeploy.model import MODELS


def get_tp_num(config, model):
    tp_config = config.get('tp_config')
    tp_num = 1
    if tp_config is None:
        return None
    model_name = _simple_model_name(model)
    if model_name in tp_config.keys():
        tp_num = tp_config.get(model_name)
    return tp_num


def get_model_name(model):
    model_names = [
        'llama', 'llama2', 'llama3', 'internlm', 'internlm2'
    ]
    model_names += list(MODELS.module_dict.keys())
    model_names.sort()
    model_name = _simple_model_name(model)
    model_name = model_name.lower()

    if model_name in model_names:
        return model_name
    if model_name in model_names:
        return model_name
    if ('llama-2' in model_name):
        return 'llama2'
    if ('internlm2') in model_name:
        return 'internlm2'
    if len(model_name.split('-')) > 2 and '-'.join(
            model_name.split('-')[0:2]) in model_names:
        return '-'.join(model_name.split('-')[0:2])
    return model_name.split('-')[0]


def _simple_model_name(model):
    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model
    return model_name