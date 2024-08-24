# 介绍
dlinfer提供了一套将国产硬件接入大模型推理框架的解决方案。
对上承接大模型推理框架，对下在eager模式下调用各厂商的融合算子，在graph模式下调用厂商的图引擎。
在dlinfer中，我们根据主流大模型推理框架与主流硬件厂商的融合算子粒度，定义了大模型推理的融合算子接口。
这套融合算子接口主要功能
1. 基于这套融合算子接口，将对接框架与对接厂商融合算子在适配工程中有效解耦。
2. 同时支持图模式，使图模式下的图获取更加精确，提高最终端到端性能。
3. 同时支持大语言模型推理和多模态模型推理

目前，我们正在全力支持LMDeploy适配国产芯片，包括华为，沐曦，寒武纪等。


# 架构介绍
![dlinfer arch](assets/dlinfer_arch.png "dlinfer架构图")
## 组件介绍
- op interface
大模型推理算子接口，对齐了主流推理框架以及各个厂商的融合算子粒度。
    - 算子模式：在pytorch的eager模式下，我们将通过op interface向下分发到厂商kernel。由于各个厂商对于参数的数据排布有不同的偏好，所以在这里我们并不会规定数据排布，但是为了多硬件的统一适配，我们将会统一参数的维度信息。
    - 图模式：在极致性能的驱动下，在一些硬件上的推理场景中需要依靠图模式。我们利用统一的大模型推理算子接口，获取带有较为粗粒度算子的计算图，提供给图编译器。

- framework adaptor
将大模型推理算子接口加入推理框架中，并且对齐算子接口的参数。

- kernel adaptor
吸收了大模型推理算子接口参数和硬件厂商融合算子参数间的差异。
 

## 安装方法
### pip安装
xxx
### 源码安装
#### 华为910B
1. 在910B上依赖torch和torch_npu，运行以下命令安装torch、torch_npu及其依赖。
```
pip3 install requirements.txt --index-url https://download.pytorch.org/whl/cpu
```
2. 完成上述准备工作后，使用如下命令即可安装dlinfer。
```
cd $WORKDIR/InferExt
DEVICE=ascend python3 setup.py develop
```

## 支持模型框架列表
### LMDeploy


|  | 华为910B | 沐曦C500（待开源） | 寒武纪590（开发中） |
| --- | --- | --- | --- |
| InternLM2.5-8B | O | O |  |
| InternVL2-2B | O | O  |  |
| Llama3 | O | O  |  |
| Mixtral8x7B | O | O  |  |
| Qwen2 | O  |  O |  |
|  |  |  |  |


## Usage
### LMDeploy
只需要指定pytorch engine后端为ascend，不需要其他任何修改即可。详细可参考lmdeploy文档。
示例代码如下：
