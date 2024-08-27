# ReadMe for test model for your self

## How to add model for CI

1. 将模型权重等下载到ci机器的/data2/share_data目录(如/data2/share_data/llama_model_data/llama-2-7b-chat-hf).
2. 在config.yml中的pytorch_chat_model下添加上述模型文件夹.
3. 如果该模型的tp>1, 需要在config.yml中的tp_config下面添加"模型名：tp_num"(如Mixtral-8x7B-Instruct-v0.1: 2).

## How to run test locally

1. 修改config.yml中对应的模型路径和log_path

2. `export DLINFER_TEST_DIR=/path/to/dlinfer/tests`

3. 运行

   ```bash
   #!/bin/bash
   cd /path/to/tests
   #run tp=1 model on lmdeploy
   pytest ./ -m 'lmdeploy' -s -x --alluredir=allure-results --clean-alluredir
   #run tp=2 chat_model on lmdeploy
   python ./test_lmdeploy/e2e/test_model_tp2.py --model_type=chat --device_type=ascend
   #run tp=2 vl_model on lmdeploy
   python ./test_lmdeploy/e2e/test_model_tp2.py --model_type=vl --device_type=ascend
   ```
