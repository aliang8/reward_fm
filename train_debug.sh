#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
export RFM_DATASET_PATH=/home/thecodeboy/reward_fm/rfm_dataset

# Run training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --config_file rfm/configs/fsdp_single.yaml \
    train.py \
    --config rfm/configs/config.yaml \
    --data.resized_height 128 \
    --data.resized_width 128 \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.output_dir ./logs/rfm_debug