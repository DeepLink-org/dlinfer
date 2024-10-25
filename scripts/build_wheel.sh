#!/usr/bin/env bash
set -e

REPO_ROOT=$(cd $(dirname $(dirname $0)); pwd)
pip install -U build
rm -rf ${REPO_ROOT}/_skbuild ${REPO_ROOT}/dlinfer*.egg*
export DEVICE=${DEVICE:-ascend}
python -m build \
    -C="--build-option=--plat-name" \
    -C="--build-option=manylinux2014_$(uname -m)" \
    -v -w .
