#!/bin/bash

export GPUID=0
export NET="yolo"

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/demo.py \
  --demo_net=$NET \
  --checkpoint="/tmp/logs/+yolo+224/train/model_11999.ckpt-11999" \
  --gpu=$GPUID
