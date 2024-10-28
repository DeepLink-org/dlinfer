#!/usr/bin/env bash
set -e

eval "$(conda shell.bash hook)"

REPO_ROOT=$(cd $(dirname $(dirname $0)); pwd)
cd ${REPO_ROOT}

PY_VERSION_LIST=("3.8" "3.9" "3.10")
# PY_VERSION_LIST=("3.10")
for PY_VERSION in ${PY_VERSION_LIST[@]}; do
    echo start building wheels for python${PY_VERSION}
    PY_VERSION_NAME=${PY_VERSION/./}
    ENV_NAME=dlinfer_build_py${PY_VERSION_NAME}
    conda env remove -n ${ENV_NAME} -y
    conda create -n ${ENV_NAME} python=${PY_VERSION} -y
    conda activate ${ENV_NAME}
    pip install -U build 
    bash ${REPO_ROOT}/scripts/build_wheel.sh
    conda deactivate
    conda env remove -n ${ENV_NAME} -y
    echo end building wheels for python${PY_VERSION}
done
