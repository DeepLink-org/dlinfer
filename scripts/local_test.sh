#!/bin/bash

set -ex
# 请预先将tests/test_lmdeploy/e2e/config.yaml中的model_path和data_path修改为本地路径
# 检查参数数量是否正确
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <DLINFER_PATH> <LMDEPLOY_PATH>"
  exit 1
fi

# 设置变量
DLINFER_PATH="$1"
LMDEPLOY_PATH="$2"
LMDEPLOY_COMMIT_OR_BRANCH="main"

# 构建和安装 dlinfer
echo "Building and installing dlinfer"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd "$DLINFER_PATH"
pip uninstall -y dlinfer-ascend || true  # 忽略卸载失败的情况
pip install -r requirements/ascend/full.txt
DEVICE=ascend python3 setup.py develop --user

# 创建并清理 lmdeploy 目录
echo "Creating and cleaning directory for lmdeploy: $LMDEPLOY_PATH"
if [ -d "$LMDEPLOY_PATH" ]; then
  rm -rf "$LMDEPLOY_PATH"
fi
mkdir -p "$LMDEPLOY_PATH"

# 克隆 lmdeploy
echo "Cloning lmdeploy"
git clone https://github.com/DeepLink-org/lmdeploy.git "$LMDEPLOY_PATH"
cd "$LMDEPLOY_PATH" && git checkout "$LMDEPLOY_COMMIT_OR_BRANCH"

echo "Running tests"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="$LMDEPLOY_PATH:$PYTHONPATH"
cd "$DLINFER_PATH/tests"
export DLINFER_TEST_DIR="$DLINFER_PATH/tests"
echo $PYTHONPATH && pwd
pytest ./ -m 'lmdeploy'
bash "$DLINFER_PATH/tests/test_lmdeploy/scripts/test_model_tp2.sh"

echo "CI process completed successfully."