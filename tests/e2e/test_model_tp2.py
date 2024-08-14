import pytest

import infer_ext

from tests.utils.config_utils import get_torch_model_list, get_config, get_case_config
from tests.utils.pipeline_chat import assert_pipeline_chat_log, run_pipeline_chat_test


@pytest.mark.skip(
    reason="There is unresovalbe issue with the pytest mutil processe spwaning"
)
def test_pipeline_chat_pytorch_tp2(env_config, case_config):
    model_case_list = get_torch_model_list(tp_num=2)
    failed_result = {}
    model_passed = []
    for model_case in model_case_list:
        print("######## InferExt testting chat_model case: ", model_case)
        run_pipeline_chat_test(env_config, case_config, model_case, "pytorch")
        # assert script
        log_results = assert_pipeline_chat_log(
            env_config, case_config, model_case, "pytorch", use_pytest=False
        )
        if len(log_results) > 0:
            failed_result[model_case] = log_results
        else:
            model_passed.append(model_case)

    print("######## InferExt test passed chat_model case: ", model_passed)
    if len(failed_result) > 0:
        for model_case in failed_result.keys():
            print("######## InferExt test failed chat_model case: ", model_case)
            for case, msg in failed_result[model_case]:
                print(f"case: {case} failed, msg: {msg}")
        raise Exception("InferExt test chat_models with tp 2 failed")


if __name__ == "__main__":
    env_config = get_config()
    case_config = get_case_config()
    test_pipeline_chat_pytorch_tp2(env_config, case_config)
