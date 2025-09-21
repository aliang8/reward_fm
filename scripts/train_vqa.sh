#!/bin/bash

CUDA_VISIBLE_DEVICES=6 uv run accelerate launch \
    --config_file rfm/configs/fsdp_single.yaml \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --training.output_dir ./logs/rfm_vqa