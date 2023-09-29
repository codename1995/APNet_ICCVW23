##!/bin/bash
PYTHON="/home/username/.conda/envs/APNet/bin/python"
GPU_NUM=8
CONFIG="experiments/sensaturban/apnet.yaml"

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        --master_port 66668 \
        train_SensatUrban.py \
        --cfg $CONFIG \
        2>&1 | tee local_log.txt