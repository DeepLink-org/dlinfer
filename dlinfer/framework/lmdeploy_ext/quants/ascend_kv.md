
# KV Cache量化

目前在华为Atlas 800T A2设备，由于算子功能限制，在算子模式下，仅支持离线量化。

## KV Cache量化前提

- **依赖**

```shell
torch==2.1.0
torchvision==0.16.0
torch-npu==2.1.0.post6
```

- **工具**

```shell
amct_pytorch==0.22.2(Ascend-cann-amct_8.0.RC2)
```

## KV Cache量化示例

在当前目录执行如下命令，得到量化因子记录文件，用户根据实际情况修改示例程序中的model_path和dataset_path，并根据模型结构修改quant_layers。

```shell
python3 ascend_kv.py
```

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./outputs文件夹，该文件夹内包含以下内容：

- **config.json**：量化配置文件，描述了如何对模型中的每一层进行量化。
- **record.txt**：量化因子记录文件。

用户在使用lmdeploy时，通过环境变量ASCEND_QUANT_RECORD_FILE指定量化因子路径，并通过参数quant_policy=8，即可使用量化因子记录文件完成推理。
示例代码如下：

```python
import lmdeploy
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    pipe = lmdeploy.pipeline("/path_to_model",
                            backend_config = PytorchEngineConfig(tp=1,
                            cache_max_entry_count=0.4, device_type="ascend",
                            eager_mode=True, quant_policy=8))
    question = ["Shanghai is", "Please introduce China", "How are you?"]
    response = pipe(question, request_output_len=256, do_preprocess=False)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()
```
