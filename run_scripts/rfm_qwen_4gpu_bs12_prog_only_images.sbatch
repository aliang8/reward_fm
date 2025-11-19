#!/bin/bash
#SBATCH -J ant_rfm_qwen_prog_only_images
#SBATCH --gres=gpu:4               # number of GPUs to reserve
#SBATCH --cpus-per-task=32         # CPUs for data loading (8 per GPU is reasonable)
#SBATCH --mem=128G                 # Memory allocation (adjust based on your cluster)
#SBATCH --time=24:00:00            # Time limit (adjust as needed)
#SBATCH --output=slurm_logs/slurm_%j.out # Standard output
#SBATCH --error=slurm_logs/slurm_%j.err  # Standard error

# Exit on error
set -e

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set environment variables

export WANDB_PROJECT=rfm
export WANDB_ENTITY=clvr
BS=8

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for progress-only training
uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=1 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/data/oxe_mw.yaml \
    --data.sample_type_ratio '[0, 1, 0]' \
    --data.progress_strategy_ratio '[1, 5, 1, 1]' \
    --model.train_progress_head true \
    --model.train_preference_head false \
    --model.train_similarity_head false \
    --model.train_success_head false \
    --model.train_language_model true \
    --model.train_vision_encoder false \
    --model.average_temporal_patches true \
    --model.base_model_id Qwen/Qwen3-VL-4B-Instruct \
    --model.use_unsloth true \
    --loss.predict_last_frame_progress false \
    --data.resized_height 196 \
    --data.resized_width 196 \
    --data.use_multi_image true \
    --data.shuffle_progress_frames false \
    --training.max_steps 15000 \
    --training.eval_steps 200 \
    --training.custom_eval_steps 200 \
    --training.per_device_train_batch_size $BS \
    --training.per_device_eval_batch_size $BS \
    --training.learning_rate 5e-5 \
    --training.weight_decay 0.05 \
    --training.overwrite_output_dir true \
    --training.exp_name ant_rfm_qwen3_prog_only_images_bs12_prog \
    --logging.log_to '["wandb"]'

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"


# CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
#     --config_file rfm/configs/distributed/ddp.yaml \
#     --num_processes=1 \
#     train.py \
#     --config_paths rfm/configs/config.yaml rfm/configs/data/oxe_mw.yaml \
#     --data.sample_type_ratio '[0, 1, 0]' \
#     --data.progress_strategy_ratio '[1, 4, 1, 1]' \
#     --model.train_progress_head true \
#     --model.train_preference_head false \
#     --model.train_similarity_head false \
#     --model.train_success_head false \
#     --model.train_language_model true \
#     --model.train_vision_encoder false \
#     --model.average_temporal_patches true \
#     --model.base_model_id Qwen/Qwen3-VL-4B-Instruct \
#     --loss.predict_last_frame_progress true \
#     --data.resized_height 196 \
#     --data.resized_width 196 \
#     --data.use_multi_image true \
#     --training.max_steps 15000 \
#     --training.eval_steps 200 \
#     --training.custom_eval_steps 200 \
#     --training.per_device_train_batch_size 8 \
#     --training.per_device_eval_batch_size 8 \
#     --training.learning_rate 5e-5 \
#     --training.weight_decay 0.1 \
#     --training.overwrite_output_dir true \
#     --training.exp_name debug \
#     --model.use_unsloth true \
#     --debug true \
#     --logging.log_to '[]'