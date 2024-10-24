# Copyright (c) 2024, DeepLink. All rights reserved.
from multiprocessing import Process
import multiprocessing
import pytest

import dlinfer

from test_lmdeploy.utils.config_utils import get_torch_model_list
from test_lmdeploy.utils.pipeline_chat import (
    assert_pipeline_chat_log,
    run_pipeline_chat_test,
    assert_pipeline_vl_chat_log,
    run_pipeline_vl_chat_test,
)


@pytest.mark.usefixtures("common_case_config")
@pytest.mark.flaky(reruns=0)
@pytest.mark.lmdeploy
@pytest.mark.parametrize("model", get_torch_model_list(tp_num=1))
def test_pipeline_chat_pytorch_tp1_ascend(config, common_case_config, model):
    p = Process(
        target=run_pipeline_chat_test,
        args=(config, common_case_config, model, "ascend"),
    )
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, "ascend")


@pytest.mark.flaky(reruns=0)
@pytest.mark.lmdeploy
@pytest.mark.parametrize("model", get_torch_model_list(tp_num=1, model_type="vl_model"))
def test_pipeline_vl_pytorch_tp1_ascend(config, model):
    p = Process(target=run_pipeline_vl_chat_test, args=(config, model, "ascend"))
    p.start()
    p.join()

    # assert script
    assert_pipeline_vl_chat_log(config, model, "ascend")
