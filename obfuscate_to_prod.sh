#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -fr "${SCRIPT_DIR}/prod"

pyarmor gen -O "${SCRIPT_DIR}/prod" "${SCRIPT_DIR}/src/"
mv "${SCRIPT_DIR}/prod/pyarmor_runtime_000000/" "${SCRIPT_DIR}/prod/src/pyarmor_runtime_000000/"
cp -r "${SCRIPT_DIR}/pretrained-models/" "${SCRIPT_DIR}/prod/pretrained-models/"
cp "${SCRIPT_DIR}/requirements_lin.txt" "${SCRIPT_DIR}/prod/requirements_lin.txt"
cp "${SCRIPT_DIR}/run.sh" "${SCRIPT_DIR}/prod/run.sh"
cp "${SCRIPT_DIR}/LISEZMOI.md" "${SCRIPT_DIR}/prod/LISEZMOI.md"
#rm -r "${SCRIPT_DIR}/prod/src/utils"
rm "${SCRIPT_DIR}/prod/src/models/coco_eval.py"
rm "${SCRIPT_DIR}/prod/src/models/dataset.py"
rm "${SCRIPT_DIR}/prod/src/models/engine.py"
rm "${SCRIPT_DIR}/prod/src/models/test.py"
rm "${SCRIPT_DIR}/prod/src/models/tracking.py"
rm "${SCRIPT_DIR}/prod/src/models/train.py"
