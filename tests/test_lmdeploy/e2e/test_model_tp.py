# Copyright (c) 2024, DeepLink. All rights reserved.
import multiprocessing
from multiprocessing import Process

import pytest

from test_lmdeploy.utils.config_utils import get_torch_model_list
from test_lmdeploy.utils.pipeline_chat import (
    assert_pipeline_chat_log,
    assert_pipeline_vl_chat_log,
    run_pipeline_chat_test,
    run_pipeline_vl_chat_test,
)
from test_lmdeploy.utils.ray_utils import cleanup_ray, join_or_kill, restart_ray_with_npu

multiprocessing.set_start_method("spawn", force=True)


@pytest.mark.usefixtures("common_case_config")
@pytest.mark.flaky(reruns=0)
@pytest.mark.lmdeploy
@pytest.mark.chat
def test_pipeline_chat_pytorch_ascend_eager(config, common_case_config, model_tp):
    model, tp = model_tp
    restart_ray_with_npu(tp)
    p = Process(
        target=run_pipeline_chat_test,
        args=(config, common_case_config, model, "ascend", True),
    )
    p.start()
    try:
        join_or_kill(p)
    finally:
        cleanup_ray()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, "ascend", True)


@pytest.mark.usefixtures("common_case_config")
@pytest.mark.flaky(reruns=0)
@pytest.mark.lmdeploy
@pytest.mark.chat
@pytest.mark.graph
def test_pipeline_chat_pytorch_ascend_graph(config, common_case_config, model_tp):
    model, tp = model_tp
    restart_ray_with_npu(tp)
    p = Process(
        target=run_pipeline_chat_test,
        args=(config, common_case_config, model, "ascend", False),
    )
    p.start()
    try:
        join_or_kill(p)
    finally:
        cleanup_ray()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, "ascend", False)


@pytest.mark.flaky(reruns=0)
@pytest.mark.lmdeploy
@pytest.mark.vl
@pytest.mark.graph
def test_pipeline_vl_pytorch_ascend_graph(config, model_tp):
    model, tp = model_tp
    restart_ray_with_npu(tp)
    p = Process(target=run_pipeline_vl_chat_test, args=(config, model, "ascend", False))
    p.start()
    try:
        join_or_kill(p)
    finally:
        cleanup_ray()

    # assert script
    assert_pipeline_vl_chat_log(config, model, "ascend", False, True)
