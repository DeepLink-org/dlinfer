# Copyright (c) 2024, DeepLink. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from test_lmdeploy.utils.config_utils import get_config, get_case_config


@pytest.fixture(scope="session")
def config():
    return get_config()


@pytest.fixture(scope="class", autouse=True)
def common_case_config():
    return get_case_config()
