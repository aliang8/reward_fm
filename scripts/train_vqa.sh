#!/bin/bash

CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --training.output_dir ./logs/rfm_vqa