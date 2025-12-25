#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=$1
CONFIG=${WORK_DIR}/r50_8x1_24e_olv2_subset_A_finetune.py
CHECKPOINT=${WORK_DIR}/latest.pth
GPUS=$2
PORT=${PORT:-28516}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out \
    --eval openlane_v2 \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log