#!/usr/bin/env bash
set -e

REPO_ROOT=$(cd $(dirname $(dirname $0)); pwd)
pip install -U build
rm -rf ${REPO_ROOT}/_skbuild ${REPO_ROOT}/dlinfer.egg* ${REPO_ROOT}/dist
export DEVICE=${DEVICE:-ascend}
python -m build -C=--plat-name=manylinux2014_aarch64 -v -w .
