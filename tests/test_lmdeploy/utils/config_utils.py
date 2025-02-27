# Copyright (c) 2024, DeepLink. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import os
import yaml

from lmdeploy.model import MODELS

TEST_DIR = os.environ.get("DLINFER_TEST_DIR")


def get_torch_model_list(
    tp_num: int = None, graph_mode: bool = False, model_type: str = "chat_model"
):
    config = get_config()
    case_list = config.get("pytorch_" + model_type)
    if tp_num is not None:
        if graph_mode:
            return [
                model
                for model in case_list
                if get_tp_num(config, model) == tp_num
                and is_graph_enabled(config, model)
            ]
        else:
            return [model for model in case_list if get_tp_num(config, model) == tp_num]
    else:
        return case_list


def get_env_with_replace_home(env_str, home_str):
    env_var = os.getenv(env_str)
    if r"${HOME}" in env_var:
        env_var = env_var.replace(r"${HOME}", home_str)
    if "$HOME" in env_var:
        env_var = env_var.replace("$HOME", home_str)
    return env_var


def get_config():
    home_dir = os.getenv("HOME")
    model_path = get_env_with_replace_home("TEST_LMDEPLOY_E2E_MODEL_PATH", home_dir)
    log_path = get_env_with_replace_home("TEST_LMDEPLOY_E2E_LOG_PATH", home_dir)
    local_pic1 = get_env_with_replace_home("TEST_LMDEPLOY_E2E_LOCAL_PIC1", home_dir)
    local_pic2 = get_env_with_replace_home("TEST_LMDEPLOY_E2E_LOCAL_PIC2", home_dir)
    assert (model_path is not None) and (log_path is not None)
    config_path = os.path.join(TEST_DIR, "test_lmdeploy/e2e/config.yaml")
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    config["model_path"] = model_path
    config["log_path"] = log_path
    os.makedirs(log_path, exist_ok=True)
    if local_pic1:
        config["LOCAL_PIC1"] = local_pic1
    if local_pic2:
        config["LOCAL_PIC2"] = local_pic2
    return config


def get_case_config():
    case_path = os.path.join(TEST_DIR, "test_lmdeploy/e2e/prompt_case.yaml")
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


def is_graph_enabled(config, model):
    graph_config = config.get("graph_config")
    graph_enabled = False
    if graph_config is None:
        return False
    model_name = _simple_model_name(model)
    if model_name in graph_config.keys():
        graph_enabled = graph_config.get(model_name)
    return graph_enabled


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
