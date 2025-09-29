#!/bin/bash

NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run python3 \
#     train.py \
#     --config rfm/configs/config.yaml \
#     --logging.use_wandb false \
#     --debug true \
#     --model.train_preference_head true \
#     --model.train_progress_head true \
#     --training.output_dir ./logs/rfm_debug

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 \
    train.py \
    --config rfm/configs/config.yaml \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.output_dir ./logs/rfm_test