# 介绍
Deeplink-Infer提供了一套将国产硬件接入大模型推理框架的解决方案。
对上承接大模型推理框架，对下在eager模式下调用各厂商的融合算子，在graph模式下调用厂商的图引擎。
在Deeplink-Infer中，我们根据主流大模型推理框架与主流硬件厂商的融合算子粒度，定义了大模型推理的融合算子接口。
这套融合算子接口主要功能
1. 基于这套融合算子接口，将对接框架与对接厂商融合算子在适配工程中有效解耦。
2. 同时支持图模式，使图模式下的图获取更加精确，提高最终性能。
3. 同时支持大语言模型推理和多模态模型推理


# 架构介绍

- framework adaptor
- op interface
- kernel adaptor

## Install
### ascend
InferExt在910B上依赖torch和torch_npu，运行以下命令安装torch、torch_npu及其依赖。
```
pip3 install requirements.txt --index-url https://download.pytorch.org/whl/cpu

```
完成上述准备工作后，使用如下命令即可安装InferExt。
```
cd $WORKDIR/InferExt
DEVICE=ascend python3 setup.py develop
```

## Usage
参考如下代码，即可实现在推理时使用InferExt的扩展算子。
```
import infer_ext.ops as ext_ops
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int):
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(
        router_logits, topk)
    return routing_weights, selected_experts
```
