import os
import yaml

from ci_utils.get_run_config import get_tp_num


def get_torch_model_list(tp_num: int = None, model_type: str = 'chat_model'):
    config = get_config()
    case_list = config.get('pytorch_' + model_type)
    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list

def get_config():
    config_path = os.path.join('./config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config