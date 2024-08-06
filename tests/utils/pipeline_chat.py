import os
import subprocess
from subprocess import PIPE

import allure
import torch
from pytest import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig
from lmdeploy import PytorchEngineConfig

import infer_ext

def run_pipeline_chat_test(config,
                           cases_info,
                           model_case,
                           type,
                           worker_id: str = '',
                           extra: object = None,
                           use_local_model: bool = True):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_name = model_name = get_model_name(model_case)
    model_path = config.get('model_path')
    if use_local_model is True:
        hf_path = model_path + '/' + model_case
    else:
        hf_path = model_case
    # InferExt is only for ascend backend for now, we can add more backend support by exposing device_type arg
    backend_config = PytorchEngineConfig(tp=tp, device_type="ascend")

    pipe = pipeline(hf_path, backend_config=backend_config)

    # run testcases
    gen_config = GenerationConfig(top_k=1)

    config_log = os.path.join(
        log_path, '_'.join([
            'pipeline', 'config', type, worker_id,
            model_case.split('/')[1] + '.log'
        ]))
    file = open(config_log, 'w')
    log_string = '\n'.join([
        'reproduce config info:', 'engine_config = ' + str(backend_config),
        'gen_config = ' + str(gen_config),
        'pipe = pipeline("' + hf_path + '",  backend_config=engine_config)',
        'res = pipe("Hi, pls introduce shanghai", gen_config=gen_config)'
    ])
    file.writelines(log_string)
    print("log config: ", log_string)
    file.close

    for case in cases_info.keys():
        if ('coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path, '_'.join([
                'pipeline', 'chat', type, worker_id,
                model_case.split('/')[1], case + '.log'
            ]))

        file = open(pipeline_chat_log, 'w')

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            file.writelines('prompt:' + prompt + '\n')

            response = pipe([prompts], gen_config=gen_config)[0].text

            case_result, reason = assert_result(response,
                                                prompt_detail.values(),
                                                model_name)
            prompts.append({'role': 'assistant', 'content': response})
            file.writelines('output:' + response + '\n')
            file.writelines('result:' + str(case_result) + ', reason:' +
                            reason + '\n')
        file.close()

    del pipe
    # TODO. is it suitable for NPU?
    torch.cuda.empty_cache()


def assert_pipeline_chat_log(config,
                             cases_info,
                             model_case,
                             type,
                             worker_id: str = ''):
    log_path = config.get('log_path')

    config_log = os.path.join(
        log_path, '_'.join([
            'pipeline', 'config', type, worker_id,
            model_case.split('/')[1] + '.log'
        ]))

    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)

    for case in cases_info.keys():
        if ('coder' in model_case
                or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        msg = 'result is empty, please check again'
        result = False
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, '_'.join([
                    'pipeline', 'chat', type, worker_id,
                    model_case.split('/')[1], case + '.log'
                ]))

            allure.attach.file(pipeline_chat_log,
                               attachment_type=allure.attachment_type.TEXT)

            with open(pipeline_chat_log, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if 'result:False, reason:' in line:
                        result = False
                        msg = line
                        break
                    if 'result:True, reason:' in line and result is False:
                        result = True
                        msg = ''

            with assume:
                assert result, msg