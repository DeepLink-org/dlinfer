# Copyright (c) 2024, DeepLink. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from test_lmdeploy.utils.config_utils import get_torch_model_list
from test_lmdeploy.utils.config_utils import get_config, get_case_config


@pytest.fixture(scope="session")
def config():
    return get_config()


@pytest.fixture(scope="class", autouse=True)
def common_case_config():
    return get_case_config()


TP_VALUES = [1, 2]


def pytest_generate_tests(metafunc):
    if "model_tp" in metafunc.fixturenames:
        if metafunc.definition.get_closest_marker("chat"):
            model_type = "chat_model"
        elif metafunc.definition.get_closest_marker("vl"):
            model_type = "vl_model"
        else:
            return

        graph_mode = False
        if metafunc.definition.get_closest_marker("graph"):
            graph_mode = True

        combos = []
        for tp in TP_VALUES:
            models = get_torch_model_list(
                tp_num=tp, graph_mode=graph_mode, model_type=model_type
            )
            for model in models:
                combos.append((model, tp))
        metafunc.parametrize("model_tp", combos, ids=lambda x: f"{x[0]}_tp{x[1]}")
