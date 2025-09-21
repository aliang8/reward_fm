#!/bin/bash

NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1
USE_ACCELERATE=true
OUTPUT_DIR=./logs/rfm_prefprog

if [ "${USE_ACCELERATE}" = true ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run accelerate launch \
        --config_file rfm/configs/fsdp.yaml \
        --num_processes=${NUM_GPUS} \
        train.py \
        --config rfm/configs/config.yaml \
        --logging.use_wandb true \
        --debug false \
        --model.train_preference_head true \
        --model.train_progress_head true \
        --training.output_dir ${OUTPUT_DIR} 
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run python train.py \
        --config rfm/configs/config.yaml \
        --logging.use_wandb true \
        --debug false \
        --model.train_preference_head true \
        --model.train_progress_head true \
        --training.output_dir ${OUTPUT_DIR} 
fi
