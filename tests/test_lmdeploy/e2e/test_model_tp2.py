import pytest
import argparse

import dlinfer

from test_lmdeploy.utils.config_utils import get_torch_model_list, get_config, get_case_config
from test_lmdeploy.utils.pipeline_chat import (
    assert_pipeline_chat_log,
    run_pipeline_chat_test,
    assert_pipeline_vl_chat_log,
    run_pipeline_vl_chat_test,
)


@pytest.mark.skip(
    reason="There is unresolvable issue with the pytest multi process spawning"
)
def test_pipeline_chat_pytorch_tp2(env_config, case_config, device_type):
    model_case_list = get_torch_model_list(tp_num=2)
    failed_result = {}
    model_passed = []
    print(f"######## dlinfer testting chat_models on {device_type}")
    for model_case in model_case_list:
        print("######## dlinfer testting chat_model case: ", model_case)
        run_pipeline_chat_test(env_config, case_config, model_case, device_type)
        # assert script
        log_results = assert_pipeline_chat_log(
            env_config, case_config, model_case, device_type, use_pytest=False
        )
        if len(log_results) > 0:
            failed_result[model_case] = log_results
        else:
            model_passed.append(model_case)

    print("######## dlinfer test passed chat_model case: ", model_passed)
    if len(failed_result) > 0:
        for model_case in failed_result.keys():
            print("######## dlinfer test failed chat_model case: ", model_case)
            for case, msg in failed_result[model_case]:
                print(f"case: {case} failed, msg: {msg}")
        raise Exception(f"dlinfer test chat_models with tp2 on {device_type} failed")


@pytest.mark.skip(
    reason="There is unresolvable issue with the pytest multi process spawning"
)
def test_pipeline_vl_pytorch_tp2(env_config, device_type):
    model_case_list = get_torch_model_list(tp_num=2, model_type="vl_model")
    failed_result = {}
    model_passed = []
    print(f"######## dlinfer testting chat_models on {device_type}")
    for model_case in model_case_list:
        print("######## dlinfer testting vl_model case: ", model_case)
        run_pipeline_vl_chat_test(env_config, model_case, device_type)
        # assert script
        log_results = assert_pipeline_vl_chat_log(
            env_config, model_case, device_type, use_pytest=False
        )
        if len(log_results) > 0:
            failed_result[model_case] = log_results
        else:
            model_passed.append(model_case)

    print("######## dlinfer test passed vl_model case: ", model_passed)
    if len(failed_result) > 0:
        for model_case in failed_result.keys():
            print("######## dlinfer test failed vl_model case: ", model_case)
            for result, msg in failed_result[model_case]:
                print(f"result: {result}, msg: {msg}")
        raise Exception(f"dlinfer test vl_models with tp2 on {device_type} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["chat", "vl"], required=True)
    parser.add_argument("--device_type", choices=["ascend"], required=True)
    args = parser.parse_args()
    env_config = get_config()
    case_config = get_case_config()
    if args.model_type == "chat":
        test_pipeline_chat_pytorch_tp2(env_config, case_config, args.device_type)
    elif args.model_type == "vl":
        test_pipeline_vl_pytorch_tp2(env_config, args.device_type)