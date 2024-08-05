import os

import yaml
from utils.get_run_config import get_tp_num


def get_torch_model_list(tp_num: int = None, model_type: str = 'chat_model'):
    config = get_config()
    case_list = config.get('pytorch_' + model_type)
    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_vl_model_list(tp_num: int = None):
    config = get_config()

    case_list = config.get('vl_model')

    for key in config.get('quatization_case_config').get('w4a16'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return 'CUDA_VISIBLE_DEVICES=' + worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return 'CUDA_VISIBLE_DEVICES=' + ','.join(
                [str(cuda_num), str(cuda_num + 1)])


def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])
        elif tp_num == 4:
            cuda_num = int(worker_id.replace('gw', '')) * 4
            return ','.join([
                str(cuda_num),
                str(cuda_num + 1),
                str(cuda_num + 2),
                str(cuda_num + 3)
            ])


def get_config():
    config_path = os.path.join('tests/config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))
