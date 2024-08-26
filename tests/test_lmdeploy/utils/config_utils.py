import os
import yaml

from lmdeploy.model import MODELS

TEST_DIR = os.environ.get("DLINFER_TEST_DIR")


def get_torch_model_list(tp_num: int = None, model_type: str = "chat_model"):
    config = get_config()
    case_list = config.get("pytorch_" + model_type)
    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_config():
    config_path = os.path.join(TEST_DIR + "/test_lmdeploy/e2e/config.yaml")
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def get_case_config():
    case_path = os.path.join(TEST_DIR + "/test_lmdeploy/e2e/prompt_case.yaml")
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


def get_tp_num(config, model):
    tp_config = config.get("tp_config")
    tp_num = 1
    if tp_config is None:
        return None
    model_name = _simple_model_name(model)
    if model_name in tp_config.keys():
        tp_num = tp_config.get(model_name)
    return tp_num


def get_model_name(model):
    model_names = ["llama", "llama2", "llama3", "internlm", "internlm2"]
    model_names += list(MODELS.module_dict.keys())
    model_names.sort()
    model_name = _simple_model_name(model)
    model_name = model_name.lower()

    if model_name in model_names:
        return model_name
    if model_name in model_names:
        return model_name
    if "llama-2" in model_name:
        return "llama2"
    if "llama-3" in model_name:
        return "llama3"
    if ("internlm2") in model_name:
        return "internlm2"
    if (
        len(model_name.split("-")) > 2
        and "-".join(model_name.split("-")[0:2]) in model_names
    ):
        return "-".join(model_name.split("-")[0:2])
    return model_name.split("-")[0]


def _simple_model_name(model):
    if "/" in model:
        model_name = model.split("/")[1]
    else:
        model_name = model
    return model_name
