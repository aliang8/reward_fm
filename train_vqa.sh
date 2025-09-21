#!/bin/bash

export RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset
export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets

# Run training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=6 uv run accelerate launch \
    --config_file rfm/configs/fsdp_single.yaml \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --training.output_dir ./logs/rfm_vqa