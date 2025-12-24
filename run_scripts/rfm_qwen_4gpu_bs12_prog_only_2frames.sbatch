#!/bin/bash
#SBATCH -J ant_rfm_qwen_2gpu_bs12_prog_only_2frames
#SBATCH --gres=gpu:2               # number of GPUs to reserve
#SBATCH --cpus-per-task=8         # CPUs for data loading (8 per GPU is reasonable)
#SBATCH --mem=300G                 # Memory allocation (adjust based on your cluster)
#SBATCH --time=36:00:00            # Time limit (adjust as needed)
#SBATCH --output=slurm_logs/slurm_%j.out # Standard output
#SBATCH --error=slurm_logs/slurm_%j.err  # Standard error

# Exit on error
set -e

ulimit -n 65536

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set environment variables
export BS=12

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for progress-only training
uv run accelerate launch \
    --config_file rfm/configs/distributed/ddp.yaml \
    --num_processes=1 \
    train.py \
    data.sample_type_ratio=[0,1,0] \
    data.progress_strategy_ratio=[1,6,1,1,0] \
    data.train_datasets=[mw,oxe,others,paired,suboptimal_fail] \
    data.eval_datasets=[mw] \
    model.train_progress_head=true \
    model.train_preference_head=false \
    model.train_similarity_head=false \
    model.train_success_head=false \
    model.train_language_model=true \
    model.train_vision_encoder=false \
    model.average_temporal_patches=true \
    model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
    model.use_unsloth=true \
    loss.predict_last_frame_progress=false \
    data.resized_height=196 \
    data.resized_width=196 \
    data.use_multi_image=true \
    data.shuffle_progress_frames=false \
    data.dataset_type=rfm \
    data.max_frames=2 \
    data.min_frames_per_trajectory=10 \
    data.progress_pred_type="absolute_wrt_total_frames" \
    training.max_steps=15000 \
    training.eval_steps=200 \
    training.custom_eval_steps=200 \
    logging.save_best.save_every=1000 \
    training.per_device_train_batch_size=$BS \
    training.per_device_eval_batch_size=$BS \
    training.learning_rate=5e-5 \
    training.weight_decay=0.05 \
    training.overwrite_output_dir=true \
    training.gradient_accumulation_steps=1 \
    training.ddp_find_unused_parameters=false \
    training.exp_name=ant_rfm_qwen_2gpu_bs12_prog_only_2frames \
    custom_eval.eval_types=[reward_alignment,policy_ranking,confusion_matrix] \
    custom_eval.reward_alignment=[reward_alignment] \
    custom_eval.policy_ranking=[policy_ranking] \
    custom_eval.confusion_matrix=[mw] \
    logging.log_level=debug \
    "logging.wandb_notes='prog only training, snoopy, 2frames with absolute progress wrt total frames'" \
    logging.log_to=[wandb]

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"