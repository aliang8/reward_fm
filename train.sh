#!/bin/bash

# Script to run RFM training with FSDP using accelerate launch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RFM_DATASET_PATH=/home/thecodeboy/reward_fm/rfm_dataset

# Run training with FSDP using accelerate launch
uv run accelerate launch \
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
    
    # --model.base_model_id Qwen/Qwen2.5-VL-7B-Instruc\B