#!/bin/bash

NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0
USE_ACCELERATE=false
EXP_NAME=rfm_progress_only

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
        --training.exp_name ${EXP_NAME} 
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run python train.py \
        --config rfm/configs/config.yaml \
        --logging.use_wandb true \
        --debug false \
        --model.train_preference_head false \
        --model.train_progress_head true \
        --training.exp_name ${EXP_NAME} 
fi


accelerate launch \
    --config_file rfm/configs/fsdp.yaml \
    --num_processes=2 \
    train.py \
    --config rfm/configs/config.yaml \
    --logging.use_wandb true \
    --debug false \
    --model.train_preference_head false \
    --model.train_progress_head true \
    --model.train_language_model true \
    --training.exp_name rfm_progress_only_mw_train_all

uv run python3 \train.py \
    --config rfm/configs/config.yaml \
    --logging.use_wandb false \
    --debug false \
    --model.train_preference_head true \
    --model.train_progress_head true \
    --model.train_language_model true \
    --training.exp_name rfm_mw_pp