#!/bin/bash

CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --training.output_dir ./logs/rewind_base_libero_mw \
    --training.predict_pref_progress false
