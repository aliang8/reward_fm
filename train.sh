#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
export RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset
export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets

# Run training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
    --config_file rfm/configs/fsdp.yaml \
    train.py \
    --config rfm/configs/config.yaml \
    --data.resized_height 128 \
    --data.resized_width 128 \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.output_dir ./logs/rfm_prefprog 
