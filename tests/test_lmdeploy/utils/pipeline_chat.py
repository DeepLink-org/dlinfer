# Copyright (c) 2024, DeepLink. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import os
import allure
import torch

from pytest_assume.plugin import assume

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from .config_utils import get_model_name, get_tp_num
from .rule_condition_assert import assert_result


def run_pipeline_chat_test(
    config,
    cases_info,
    model_case,
    device_type,
    extra: object = None,
    use_local_model: bool = True,
):
    log_path = config.get("log_path")
    tp = get_tp_num(config, model_case)
    model_name = model_name = get_model_name(model_case)
    model_path = config.get("model_path")
    if use_local_model is True:
        hf_path = model_path + "/" + model_case
    else:
        hf_path = model_case
    backend_config = PytorchEngineConfig(tp=tp, device_type=device_type)
    print("backend_config: ", backend_config)
    pipe = pipeline(hf_path, backend_config=backend_config)

    # run testcases
    gen_config = GenerationConfig(top_k=1)

    config_log = os.path.join(
        log_path,
        "_".join(
            [
                device_type,
                "pipeline",
                "config",
                "pytorch",
                model_case.split("/")[1] + ".log",
            ]
        ),
    )
    file = open(config_log, "w")
    log_string = "\n".join(
        [
            "reproduce config info:",
            "engine_config = " + str(backend_config),
            "gen_config = " + str(gen_config),
            'pipe = pipeline("' + hf_path + '",  backend_config=engine_config)',
            'res = pipe("Hi, pls introduce shanghai", gen_config=gen_config)',
        ]
    )
    file.writelines(log_string)
    print("log config: ", log_string)
    file.close

    for case in cases_info.keys():
        if ("coder" in model_case or "CodeLlama" in model_case) and "code" not in case:
            continue
        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path,
            "_".join(
                [
                    device_type,
                    "pipeline",
                    "chat",
                    "pytorch",
                    model_case.split("/")[1],
                    case + ".log",
                ]
            ),
        )

        file = open(pipeline_chat_log, "w")

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({"role": "user", "content": prompt})
            file.writelines("prompt:" + prompt + "\n")

            response = pipe([prompts], gen_config=gen_config)[0].text

            case_result, reason = assert_result(
                response, prompt_detail.values(), model_name
            )
            prompts.append({"role": "assistant", "content": response})
            file.writelines("output:" + response + "\n")
            file.writelines("result:" + str(case_result) + ", reason:" + reason + "\n")
        file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_chat_log(
    config, cases_info, model_case, device_type, use_pytest=True
):
    log_path = config.get("log_path")

    config_log = os.path.join(
        log_path,
        "_".join(
            [
                device_type,
                "pipeline",
                "config",
                "pytorch",
                model_case.split("/")[1] + ".log",
            ]
        ),
    )

    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)
    log_results = []
    for case in cases_info.keys():
        if ("coder" in model_case or "CodeLlama" in model_case) and "code" not in case:
            continue
        msg = "result is empty, please check again"
        result = False
        with allure.step("case - " + case):
            pipeline_chat_log = os.path.join(
                log_path,
                "_".join(
                    [
                        device_type,
                        "pipeline",
                        "chat",
                        "pytorch",
                        model_case.split("/")[1],
                        case + ".log",
                    ]
                ),
            )

            allure.attach.file(
                pipeline_chat_log, attachment_type=allure.attachment_type.TEXT
            )

            with open(pipeline_chat_log, "r") as f:
                lines = f.readlines()

                for line in lines:
                    if "result:False, reason:" in line:
                        result = False
                        msg = line
                        break
                    if "result:True, reason:" in line and result is False:
                        result = True
                        msg = ""
            if use_pytest:
                with assume:
                    assert result, msg
            else:
                if not result:
                    log_results.append((case, msg))
    return log_results


PIC1 = (
    "https://raw.githubusercontent.com/"
    + "open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
)
PIC2 = (
    "https://raw.githubusercontent.com/"
    + "open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg"
)


def run_pipeline_vl_chat_test(config, model_case, device_type):
    log_path = config.get("log_path")
    tp = get_tp_num(config, model_case)
    model_path = config.get("model_path")
    hf_path = model_path + "/" + model_case
    backend_config = PytorchEngineConfig(
        tp=tp, session_len=8192, device_type=device_type
    )
    pipe = pipeline(hf_path, backend_config=backend_config)

    pipeline_chat_log = os.path.join(
        log_path, device_type + "_pipeline_vl_chat_" + model_case.split("/")[1] + ".log"
    )
    file = open(pipeline_chat_log, "w")

    image = load_image(PIC1)
    prompt = "describe this image"
    response = pipe((prompt, image))
    result = "tiger" in response.text.lower() or "虎" in response.text.lower()
    file.writelines(
        "result:"
        + str(result)
        + ", reason: simple example tiger not in "
        + response.text
        + "\n"
    )

    prompts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": PIC1}},
            ],
        }
    ]
    response = pipe(prompts)
    result = "tiger" in response.text.lower() or "虎" in response.text.lower()
    file.writelines(
        "result:"
        + str(result)
        + ", reason: OpenAI format example: tiger not in "
        + response.text
        + "\n"
    )

    image_urls = [PIC2, PIC1]
    images = [load_image(img_url) for img_url in image_urls]
    # test for multi batchs
    response = pipe((prompt, images))
    result = (
        "tiger" in response.text.lower()
        or "ski" in response.text.lower()
        or "虎" in response.text.lower()
        or "滑雪" in response.text.lower()
    )
    file.writelines(
        "result:"
        + str(result)
        + ", reason: Multi-images example: tiger or ski not in "
        + response.text
        + "\n"
    )

    image_urls = [PIC2, PIC1]
    prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts)
    result = (
        "ski" in response[0].text.lower() or "滑雪" in response[0].text.lower()
    ) and ("tiger" in response[1].text.lower() or "虎" in response[1].text.lower())
    file.writelines(
        "result:"
        + str(result)
        + ", reason: Batch example: ski or tiger not in "
        + str(response)
        + "\n"
    )
    # test for conversation
    image = load_image(PIC2)
    sess = pipe.chat((prompt, image))
    result = "ski" in sess.response.text.lower() or "滑雪" in sess.response.text.lower()
    file.writelines(
        "result:"
        + str(result)
        + ", reason: Multi-turn example: ski not in "
        + sess.response.text
        + "\n"
    )
    sess = pipe.chat("What is the woman doing?", session=sess)
    result = "ski" in sess.response.text.lower() or "滑雪" in sess.response.text.lower()
    file.writelines(
        "result:"
        + str(result)
        + ", reason: Multi-turn example: ski not in "
        + sess.response.text
        + "\n"
    )

    file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_vl_chat_log(config, model_case, device_type, use_pytest=True):
    log_path = config.get("log_path")

    pipeline_chat_log = os.path.join(
        log_path, device_type + "_pipeline_vl_chat_" + model_case.split("/")[1] + ".log"
    )

    allure.attach.file(pipeline_chat_log, attachment_type=allure.attachment_type.TEXT)

    msg = "result is empty, please check again"
    result = False
    with open(pipeline_chat_log, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "result:False, reason:" in line:
                result = False
                msg = line
                break
            if "result:True, reason:" in line and result is False:
                result = True
                msg = ""

    log_results = []
    if use_pytest:
        with assume:
            assert result, msg
    else:
        if not result:
            log_results.append((result, msg))
    return log_results
