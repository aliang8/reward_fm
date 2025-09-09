#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0
export RFM_DATASET_PATH=/home/thecodeboy/reward_fm/rfm_dataset

# Run training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --data.resized_height 128 \
    --data.resized_width 128 \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.output_dir ./logs/rewind_debug