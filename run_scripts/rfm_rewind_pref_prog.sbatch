#!/bin/bash
#SBATCH -J ant_rfm_rewind_bs1024_pref_prog
#SBATCH --gres=gpu:1               # number of GPUs to reserve
#SBATCH --cpus-per-task=8         # CPUs for data loading (8 per GPU is reasonable)
#SBATCH --mem=64G                 # Memory allocation (adjust based on your cluster)
#SBATCH --time=5:00:00            # Time limit (adjust as needed)
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
export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets/
export BS=1024

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for progress-only training
uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=1 \
    train.py \
    --config-name=rewind_transformer \
    data.sample_type_ratio=[1,0,0] \l
    model.train_preference_head=true \
    model.train_progress_head=true \
    model.train_success_head=false \
    data.train_datasets=[mw,others] \
    data.eval_datasets=[mw] \
    data.dataset_type=rfm \
    training.max_steps=20000 \
    training.eval_steps=200 \
    training.custom_eval_steps=200 \
    training.per_device_train_batch_size=$BS \
    training.per_device_eval_batch_size=$BS \
    training.overwrite_output_dir=true \
    training.predict_pref_progress=true \
    training.exp_name=ant_rfm_rewind_bs1024_pref_prog \
    custom_eval.eval_types=[quality_preference,reward_alignment,policy_ranking,confusion_matrix] \
    custom_eval.quality_preference=[quality_preference] \
    custom_eval.reward_alignment=[reward_alignment] \
    custom_eval.policy_ranking=[policy_ranking] \
    custom_eval.confusion_matrix=[mw] \
    logging.log_to=[wandb]

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"



uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=1 \
    train.py \
    --config-name=rewind_transformer \
    data.sample_type_ratio=[0,1,0] \
    model.train_preference_head=false \
    model.train_progress_head=true \
    model.train_success_head=false \
    data.train_datasets=[mw,oxe,others] \
    data.eval_datasets=[mw] \
    data.dataset_type=data_source_balance \
    data.data_source_weights.metaworld_train=100 \
    training.max_steps=20000 \
    training.eval_steps=200 \
    training.custom_eval_steps=200 \
    training.per_device_train_batch_size=$BS \
    training.per_device_eval_batch_size=$BS \
    training.overwrite_output_dir=true \
    training.predict_pref_progress=true \
    training.exp_name=ant_rfm_rewind_bs1024_prog_balanced \
    custom_eval.eval_types=[quality_preference,reward_alignment,policy_ranking,confusion_matrix] \
    custom_eval.quality_preference=[quality_preference] \
    custom_eval.reward_alignment=[reward_alignment] \
    custom_eval.policy_ranking=[policy_ranking] \
    custom_eval.confusion_matrix=[mw] \
    logging.log_to=[wandb]