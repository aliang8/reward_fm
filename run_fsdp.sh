#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0,1

# Run training with FSDP using accelerate launch
accelerate launch \
    --config_file rfm/configs/fsdp.yaml \
    train.py \
    --config rfm/configs/config.yaml \
    --data.resized_height 128 \
    --data.resized_width 128 \
    --data.max_frames 4 \
    --data.force_reprocess true