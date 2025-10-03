#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.exp_name vqa_debug