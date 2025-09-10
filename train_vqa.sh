#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
export RFM_DATASET_PATH=/home/thecodeboy/reward_fm/rfm_dataset

# Run training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run accelerate launch \
    --config_file rfm/configs/fsdp.yaml \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --data.resized_height 128 \
    --data.resized_width 128 \
    --logging.use_wandb true \
    --debug false \
    --training.output_dir ./logs/rfm_vqa