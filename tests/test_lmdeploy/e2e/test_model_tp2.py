# Copyright (c) 2024, DeepLink. All rights reserved.
import pytest
import argparse

import dlinfer

from test_lmdeploy.utils.config_utils import (
    get_torch_model_list,
    get_config,
    get_case_config,
)
from test_lmdeploy.utils.pipeline_chat import (
    assert_pipeline_chat_log,
    run_pipeline_chat_test,
    assert_pipeline_vl_chat_log,
    run_pipeline_vl_chat_test,
)

@pytest.mark.skip(
    reason="There is unresolvable issue with the pytest multi process spawning"
)
@pytest.mark.lmdeploy
def test_pipeline_chat_pytorch_tp2(model_case, env_config, case_config, device_type):
    print("######## dlinfer testting chat_model case: ", model_case)
    run_pipeline_chat_test(env_config, case_config, model_case, device_type)
    # assert script
    log_results = assert_pipeline_chat_log(
        env_config, case_config, model_case, device_type, use_pytest=False
    )
    if len(log_results) > 0:
        for case, msg in log_results:
            print(f"case: {case} failed, msg: {msg}")
        raise Exception(f"dlinfer test {model_case} with tp2 on {device_type} failed")
    else:
        print(f"dlinfer test {model_case} with tp2 on {device_type} passed")
    
@pytest.mark.skip(
    reason="There is unresolvable issue with the pytest multi process spawning"
)
@pytest.mark.lmdeploy
def test_pipeline_vl_pytorch_tp2(model_case, env_config, device_type):
    print("######## dlinfer testting vl_model case: ", model_case)
    run_pipeline_vl_chat_test(env_config, model_case, device_type)
    # assert script
    log_results = assert_pipeline_vl_chat_log(
        env_config, model_case, device_type, use_pytest=False
    )
    if len(log_results) > 0:
        for result, msg in log_results:
            print(f"result: {result}, msg: {msg}")
        raise Exception(f"dlinfer test {model_case} with tp2 on {device_type} failed")
    else:
        print(f"dlinfer test {model_case} with tp2 on {device_type} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_case", required=True)
    parser.add_argument("--model_type", choices=["chat", "vl"], required=True)
    parser.add_argument("--device_type", choices=["ascend"], required=True)
    args = parser.parse_args()
    env_config = get_config()
    case_config = get_case_config()
    if args.model_type == "chat":
        test_pipeline_chat_pytorch_tp2(args.model_case, env_config, case_config, args.device_type)
    elif args.model_type == "vl":
        test_pipeline_vl_pytorch_tp2(args.model_case, env_config, args.device_type)
