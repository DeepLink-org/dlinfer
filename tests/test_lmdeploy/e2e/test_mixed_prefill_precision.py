import gc
import os
from pathlib import Path

import pytest
import torch

from lmdeploy import GenerationConfig, PytorchEngineConfig, Tokenizer, pipeline
from ..utils.ray_utils import cleanup_ray, restart_ray_with_npu

ANSWER_TAG_SEEDS = (
    "OK_314159",
    "OK_271828",
    "OK_161803",
    "OK_141421",
    "OK_173205",
    "OK_223606",
    "OK_244949",
    "OK_264575",
)


def _resolve_case_config(config):
    case_config = config.get("mixed_prefill_precision")
    if case_config is None:
        raise ValueError(
            "missing mixed_prefill_precision config in test_lmdeploy/e2e/config.yaml"
        )
    prompt_token_lengths = case_config.get("prompt_token_lengths")
    if not isinstance(prompt_token_lengths, list) or len(prompt_token_lengths) == 0:
        raise ValueError(
            "mixed_prefill_precision.prompt_token_lengths must be a non-empty list"
        )
    if len(prompt_token_lengths) > len(ANSWER_TAG_SEEDS):
        raise ValueError(
            f"prompt_token_lengths length {len(prompt_token_lengths)} exceeds supported answer tag count {len(ANSWER_TAG_SEEDS)}"
        )
    if case_config["max_batch_size"] < len(prompt_token_lengths):
        raise ValueError(
            "max_batch_size must be greater than or equal to the number of prompts"
        )
    return case_config


def _resolve_model_path(model_root, model_case):
    model_path = os.path.join(model_root, model_case)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"failed to locate model path: {model_path}")
    return model_path


def _token_len(tokenizer, text):
    return len(tokenizer.encode(text, add_bos=False))


def _build_prompt(tokenizer, target_len, answer_tag):
    prefix = "下面是一些背景片段，你只需要阅读，不要总结。\n" "背景开始：\n"
    suffix = (
        "\n背景结束。\n"
        "请完成下面任务。\n"
        "你必须遵守以下规则：\n"
        "1. 不要思考过程。\n"
        "2. 不要解释。\n"
        "3. 不要重复题目。\n"
        f"4. 最终答案必须且只能是：{answer_tag}\n"
        "答案："
    )
    filler_unit = "这是一段用于混合prefill精度回归测试的背景信息。"
    fine_grained_units = [
        "补充",
        "说明",
        "数据",
        "样例",
        "文本",
        "A",
        "B",
        "C",
        "。",
        "\n",
    ]

    base_len = _token_len(tokenizer, prefix + suffix)
    if base_len > target_len:
        raise ValueError(f"base prompt length {base_len} exceeds target {target_len}")

    low = 0
    high = 1
    while _token_len(tokenizer, prefix + filler_unit * high + suffix) <= target_len:
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        prompt = prefix + filler_unit * mid + suffix
        if _token_len(tokenizer, prompt) <= target_len:
            low = mid
        else:
            high = mid

    filler = filler_unit * low
    prompt = prefix + filler + suffix

    changed = True
    while changed:
        changed = False
        for unit in fine_grained_units:
            candidate = prefix + filler + unit + suffix
            if _token_len(tokenizer, candidate) <= target_len:
                filler += unit
                prompt = candidate
                changed = True

    return prompt, _token_len(tokenizer, prompt)


def _build_prompts(model_path, case_config):
    tokenizer = Tokenizer(model_path)
    prompts = []
    actual_lengths = []
    answer_tags = _build_answer_tags(case_config["prompt_token_lengths"])
    for target_len, answer_tag in zip(case_config["prompt_token_lengths"], answer_tags):
        prompt, actual_len = _build_prompt(tokenizer, target_len, answer_tag)
        prompts.append(prompt)
        actual_lengths.append(actual_len)
    return prompts, actual_lengths, answer_tags


def _build_answer_tags(prompt_token_lengths):
    return tuple(
        f"P{idx + 1}_{ANSWER_TAG_SEEDS[idx]}"
        for idx in range(len(prompt_token_lengths))
    )


def _strip_thinking(text):
    final_text = text.strip()
    if "</think>" in final_text:
        final_text = final_text.split("</think>", 1)[1].strip()
    return final_text


@pytest.mark.lmdeploy
@pytest.mark.parametrize(
    "eager_mode",
    [True, False],
    ids=["eager", "graph"],
)
def test_mixed_prefill_precision(config, eager_mode):
    case_config = _resolve_case_config(config)
    restart_ray_with_npu(case_config["tp"])
    model_path = _resolve_model_path(config["model_path"], case_config["model_case"])
    prompts, actual_lengths, answer_tags = _build_prompts(model_path, case_config)
    model_name = Path(case_config["model_case"]).name

    backend_config = PytorchEngineConfig(
        tp=case_config["tp"],
        session_len=case_config["session_len"],
        max_batch_size=case_config["max_batch_size"],
        max_prefill_token_num=case_config["max_prefill_token_num"],
        device_type="ascend",
        eager_mode=eager_mode,
    )
    gen_config = GenerationConfig(
        do_sample=False,
        top_k=1,
        temperature=0.0,
        max_new_tokens=96,
        random_seed=0,
    )

    mode_name = "eager" if eager_mode else "graph"
    log_dir = config["log_path"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"ascend_pipeline_chat_pytorch_{model_name}_mixed_prefill_tp2_{mode_name}.log",
    )

    pipe = None
    try:
        pipe = pipeline(model_path, backend_config=backend_config)
        responses = pipe(
            prompts,
            gen_config=gen_config,
            chat_template_kwargs={"enable_thinking": False},
        )
        outputs = [response.text.strip() for response in responses]
        final_outputs = [_strip_thinking(output) for output in outputs]

        print(
            f"[mixed-prefill] model_case={case_config['model_case']} mode={mode_name}"
        )
        print(f"[mixed-prefill] prompt_token_lengths={actual_lengths}")
        for idx, (target, output, final_output) in enumerate(
            zip(answer_tags, outputs, final_outputs),
            start=1,
        ):
            print(f"[mixed-prefill] prompt_{idx}_target={target}")
            print(f"[mixed-prefill] output_{idx}={output}")
            print(f"[mixed-prefill] final_output_{idx}={final_output}")
            print(f"[mixed-prefill] match_{idx}={target in final_output}")

        with open(log_path, "w") as file:
            file.writelines(
                [
                    f"model_case: {case_config['model_case']}\n",
                    f"model_path: {model_path}\n",
                    f"backend_config: {backend_config}\n",
                    f"gen_config: {gen_config}\n",
                    f"prompt_token_lengths: {actual_lengths}\n",
                ]
            )
            for idx, (target, actual_len, prompt, output, final_output) in enumerate(
                zip(answer_tags, actual_lengths, prompts, outputs, final_outputs),
                start=1,
            ):
                file.writelines(
                    [
                        f"prompt_{idx}_target: {target}\n",
                        f"prompt_{idx}_token_len: {actual_len}\n",
                        f"prompt_{idx}_tail: {prompt[-256:]}\n",
                        f"output_{idx}: {output}\n",
                        f"final_output_{idx}: {final_output}\n",
                        f"match_{idx}: {target in final_output}\n",
                    ]
                )

        for target, output, final_output in zip(answer_tags, outputs, final_outputs):
            assert target in final_output, (
                f"mixed prefill precision regression failed in {mode_name} mode: "
                f"expected {target} in final output {final_output!r}, raw output {output!r}"
            )
    finally:
        if pipe is not None:
            pipe.close()
            del pipe
        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        cleanup_ray()
