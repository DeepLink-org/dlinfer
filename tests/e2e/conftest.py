import os

import pytest
import yaml

TEST_DIR = os.environ.get("INFEREXT_TEST_DIR")
cli_prompt_case_file = TEST_DIR + "/e2e/chat_prompt_case.yaml"
common_prompt_case_file = TEST_DIR + "/e2e/prompt_case.yaml"
config_file = TEST_DIR + "/e2e/config.yaml"


@pytest.fixture(scope="session")
def config():
    config_path = os.path.join(config_file)
    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return env_config


@pytest.fixture(scope="class", autouse=True)
def common_case_config():
    case_path = os.path.join(common_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config
