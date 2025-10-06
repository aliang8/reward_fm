#!/bin/bash

CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb false \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --training.exp_name debug \
    --training.predict_pref_progress false


accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --training.exp_name rewind_base_mw_only_more_rewind \
    --training.predict_pref_progress false

accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.exp_name rewind_base_mw_only_pp_pref_prog_relative \
    --training.predict_pref_progress true

accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --training.exp_name rewind_base_mw_relative_2

accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.exp_name debug \
    --training.predict_pref_progress false
