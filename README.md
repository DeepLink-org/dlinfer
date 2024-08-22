# dlinfer
该仓库提供了一套在国产硬件上进行大模型推理的解决方案。对下调用各厂商的PyTorch和融合算子，对上承接大模型推理框架。

## Install
### ascend
dlinfer在Atlas_800T_A2上依赖torch和torch_npu，运行以下命令安装torch、torch_npu及其依赖。
```
pip3 install requirements.txt --index-url https://download.pytorch.org/whl/cpu

```
完成上述准备工作后，使用如下命令即可安装dlinfer。
```
cd $WORKDIR/dlinfer
DEVICE=ascend python3 setup.py develop
```

## Usage
参考如下代码，即可实现在推理时使用dlinfer的扩展算子。
```
import dlinfer.ops as ext_ops
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int):
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(
        router_logits, topk)
    return routing_weights, selected_experts
```
