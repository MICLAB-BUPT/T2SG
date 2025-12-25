#!/usr/bin/env bash
set -x
export PYTHONPATH=$PYTHONPATH:"./"
timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=$1
CONFIG=projects/configs/r50_8x1_24e_olv2_subset_A.py
GPUS=$2
PORT=${PORT:-8812}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:3} \
    --auto-resume --autoscale-lr \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log