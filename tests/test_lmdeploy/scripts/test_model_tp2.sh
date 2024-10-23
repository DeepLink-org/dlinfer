#!/bin/bash

# 获取DLINFER_TEST_DIR环境变量
if [ -z "$DLINFER_TEST_DIR" ]; then
    echo "DLINFER_TEST_DIR environment variable is not set"
    exit 1
fi
echo "DLINFER_TEST_DIR: $DLINFER_TEST_DIR"

# # 获取模型列表
echo "Getting chat model list..."
chat_model_tp2_list=$(python -c "from test_lmdeploy.utils.config_utils import get_torch_model_list; print(' '.join(get_torch_model_list(tp_num=2)))")
echo "chat_model_tp2_list: $chat_model_tp2_list"

# 遍历chat模型列表
for model_case in $chat_model_tp2_list; do
    python $DLINFER_TEST_DIR/test_lmdeploy/e2e/test_model_tp2.py --model_case="$model_case" --model_type=chat --device_type=ascend
    if [ $? -ne 0 ]; then
        echo "The test for chat model $model_case failed. Exiting."
        exit 1
    fi
done

# 获取vl模型列表
echo "Getting vl model list..."
vl_model_tp2_list=$(python -c "from test_lmdeploy.utils.config_utils import get_torch_model_list; print(' '.join(get_torch_model_list(tp_num=2, model_type='vl_model')))")
echo "vl_model_tp2_list: $vl_model_tp2_list"

for model_case in $vl_model_tp2_list; do
    python $DLINFER_TEST_DIR/test_lmdeploy/e2e/test_model_tp2.py --model_case="$model_case" --model_type=vl --device_type=ascend
    if [ $? -ne 0 ]; then
        echo "The test for vl model $model_case failed. Exiting."
        exit 1
    fi
done