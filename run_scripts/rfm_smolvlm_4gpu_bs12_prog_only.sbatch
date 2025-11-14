#!/bin/bash
#SBATCH -J rfm_smolvlm_prog
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
BS=16

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project directory
cd /scr/aliang80/reward_fm

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for progress-only training
uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=4 \
    train.py \
    --config_paths rfm/configs/config.yaml \
    --data.sample_type_ratio '[0, 1, 0]' \
    --model.train_progress_head true \
    --model.train_preference_head false \
    --model.train_similarity_head false \
    --model.train_success_head false \
    --model.average_temporal_patches true \
    --data.resized_height 196 \
    --data.resized_width 196 \
    --data.use_multi_image true \
    --training.max_steps 15000 \
    --training.eval_steps 200 \
    --training.custom_eval_steps 200 \
    --training.per_device_train_batch_size $BS \
    --training.per_device_eval_batch_size $BS \
    --training.learning_rate 3e-6 \
    --training.weight_decay 0.1 \
    --training.overwrite_output_dir true \
    --training.exp_name rfm_smolvlm_prog_only_bs12_prog_only \
    --logging.use_wandb true

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"

