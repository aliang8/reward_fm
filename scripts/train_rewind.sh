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
    --training.exp_name rewind_base_mw_only_pp \
    --training.predict_pref_progress false

accelerate launch \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb false \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --training.exp_name rewind_base_mw

uv run python3 train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --training.exp_name rewind_base_mw_only_pp_2 \
    --training.predict_pref_progress false

uv run python3 train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/rewind_transformer_config.yaml \
    --logging.use_wandb false \
    --debug true \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --data.pairwise_progress true \
    --training.exp_name debug 