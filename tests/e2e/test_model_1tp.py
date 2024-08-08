import sys

from multiprocessing import Process
import pytest

import infer_ext

from ..utils.config_utils import  get_torch_model_list
from ..utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1))
def test_pipeline_chat_pytorch_tp1(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch')