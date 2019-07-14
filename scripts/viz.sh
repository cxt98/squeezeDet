#!/bin/bash

export GPUID=0
export NET="vgg16"
export OUT_DIR="./data/out/vgg16"

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/demo.py \
  --demo_net=$NET \
  --out_dir=$OUT_DIR \
  --checkpoint="/tmp/logs/+vgg16/train/model_7500.ckpt-7500" \
  --gpu=$GPUID

