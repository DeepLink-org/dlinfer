import os
import time

import tqdm
import torch
import torch.nn as nn
import dlinfer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory
from datasets import load_dataset

import amct_pytorch as amct


def get_llama2(model_path, seqlen=2048):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )

    model.seqlen = seqlen
    return model


def build_model_and_enc(model, model_path, gpu_num):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    # Move the model to GPU (as much as possible) for LM evaluation
    # max_memory = ['0:16GiB', '1:16GiB','2:16GiB', 'cpu:30GiB'], '0' means the first GPU that you specify.
    # I don't recommend use 16GiB, we need to reserve some space for other tensors during calculation
    # please see the recommand memeory allocation in the Word file
    # Adjust the max_size accroding to the real situation
    # a clever way:
    max_memory = []
    for i in range(gpu_num):
        max_memory.append(f"{i}:12GiB")
    max_memory.append("cpu:80GiB")
    print("Max_memory allocation: \n", max_memory)

    max_memory = [v.split(":") for v in (max_memory or [])]
    max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}
    kwargs = {
        "max_memory": get_balanced_memory(
            model, max_memory if len(max_memory) > 0 else None
        )
    }
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs,
    )
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=os.path.join(model_path, "offload_dir"),
    )

    return model, enc


def get_loaders(dataset_path: str, enc, seqlen):
    print("Loading dataset c4/realnewslike")
    testenc = load_dataset(
        "json", data_files={"validation": dataset_path}, split="validation"
    )
    import pdb

    pdb.set_trace()
    testenc = enc(" ".join(testenc[:1100]["text"]), return_tensors="pt")
    testenc = testenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    testenc = TokenizerWrapper(testenc)

    return testenc


def main():
    # Load model
    model_path = "/data2/share_data/internlm_model_data/internlm2_5-7b-chat"
    model = get_llama2(model_path)
    model = model.eval()
    gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    if gpus == "" or gpus is None:
        gpu_num = 0
    else:
        gpu_num = len(gpus.split(","))
    model, enc = build_model_and_enc(model, model_path, gpu_num)
    model.seqlen = 2048

    # Load dataset
    dataset_path = "./c4/c4-train.00000-of-00512.json"
    testenc = get_loaders(dataset_path=dataset_path, enc=enc, seqlen=model.seqlen)

    import pdb

    pdb.set_trace()
    testenc = testenc.input_ids.to(model.device)

    config_file = "./outputs/config.json"
    amct.create_quant_cali_config(
        config_file=config_file,
        model=model,
        quant_layers={
            "kv_cache_quant_laye": [
                f"model.layers.{i}.attention.wqkv" for i in range(32)
            ]
        },
        config_defination=None,
    )

    record_file = "./outputs/record.txt"
    quant_cali_model = amct.create_quant_cali_model(
        config_file=config_file, record_file=record_file, model=model
    ).npu()

    # Do inference to get quantize factors
    batch_num = 3
    test_start_time = time.time()
    for i in tqdm.tqdm(range(batch_num), desc="getting quantize factors..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            quant_cali_model(batch)
    test_end_time = time.time()
    total_time = test_end_time - test_start_time
    print(
        "Get quantize factors taken: ", total_time // 60, "min ", total_time % 60, "s"
    )


if __name__ == "__main__":
    main()
