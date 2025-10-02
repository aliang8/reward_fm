#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/vqa_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --training.exp_name rfm_vqa_mw 