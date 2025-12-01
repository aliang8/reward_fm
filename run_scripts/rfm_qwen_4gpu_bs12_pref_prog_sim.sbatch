#!/bin/bash
#SBATCH -J ant_rfm_qwen_4gpu_bs12_pref_prog_sim
#SBATCH --gres=gpu:2               # number of GPUs to reserve
#SBATCH --cpus-per-task=32         # CPUs for data loading (8 per GPU is reasonable)
#SBATCH --mem=128G                 # Memory allocation (adjust based on your cluster)
#SBATCH --time=18:00:00            # Time limit (adjust as needed)
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
export BS=8

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for progress-only training
uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=2 \
    train.py \
    data.sample_type_ratio=[1,0,1] \
    data.progress_strategy_ratio=[1,6,1,1,1] \
    data.preference_strategy_ratio=[6,1,1] \
    data.similarity_strategy_ratio=[1,1,1] \
    data.train_datasets=[oxe,mw,others,paired,suboptimal_fail] \
    data.eval_datasets=[mw] \
    model.train_progress_head=true \
    model.train_preference_head=true \
    model.train_similarity_head=true \
    model.train_success_head=true \
    model.train_language_model=true \
    model.train_vision_encoder=false \
    model.average_temporal_patches=true \
    model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
    model.use_unsloth=true \
    loss.predict_last_frame_progress=false \
    data.resized_height=196 \
    data.resized_width=196 \
    data.use_multi_image=true \
    data.dataset_type=rfm \
    training.max_steps=15000 \
    training.eval_steps=200 \
    training.custom_eval_steps=200 \
    training.per_device_train_batch_size=$BS \
    training.per_device_eval_batch_size=$BS \
    training.learning_rate=5e-5 \
    training.weight_decay=0.05 \
    training.overwrite_output_dir=true \
    training.predict_pref_progress=true \
    training.predict_sim_progress=true \
    training.ddp_find_unused_parameters=false \
    training.gradient_accumulation_steps=1 \
    training.exp_name=ant_rfm_qwen_4gpu_bs12_pref_prog_sim \
    custom_eval.eval_types=[quality_preference,reward_alignment,policy_ranking,confusion_matrix,similarity_score] \
    custom_eval.quality_preference=[quality_preference] \
    custom_eval.reward_alignment=[reward_alignment] \
    custom_eval.policy_ranking=[policy_ranking] \
    custom_eval.confusion_matrix=[mw] \
    custom_eval.similarity_score=[similarity_score] \
    logging.log_to=[wandb]

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"