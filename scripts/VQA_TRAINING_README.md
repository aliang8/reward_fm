# VQA Training Guide

Complete guide for generating VQA datasets and training Qwen-VL models for robotics reward modeling.

## Quick Start

```bash
# 1. Generate dataset (1 epoch = ~43k samples)
python scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 \
    --output_path ./vqa_datasets/train \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]

# 2. OR DOWNLOAD
uv run hf download rewardfm/vqa_datasets --local-dir ./vqa_datasets --repo-type dataset

# 3. Train (single GPU with Unsloth)
uv run scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/rfm_train_10epochs \
    --eval_dataset_path ./vqa_datasets/rfm_val_0.1epoch \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/vqa_training \
    --use_unsloth \
    --freeze_vision_tower \
    --lora_rank 0 \
    --learning_rate 5e-5 \
    --report_to wandb \
    --wandb_project rfm \
    --wandb_entity clvr \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_frames 32 \
    --run_name qwen3_vl_4b_vqa_train_rfm_10epochs 



# 3. Train (multi-GPU with Accelerate)
uv run torchrun --nproc_per_node=2 scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/rfm_train_10epochs \
    --eval_dataset_path ./vqa_datasets/rfm_val_0.1epoch \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/vqa_training \
    --use_unsloth \
    --freeze_vision_tower \
    --lora_rank 0 \
    --learning_rate 5e-5 \
    --report_to wandb \
    --wandb_project rfm \
    --wandb_entity clvr \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_frames 32 \
    --run_name qwen3_vl_4b_vqa_train_rfm_10epochs_multi_gpu
```

For training on roboreward dataset, change:
```bash
    --dataset_path ./vqa_datasets/rfm_train_10epochs \
    --eval_dataset_path ./vqa_datasets/rfm_val_0.1epoch 
```
---

## Dataset Generation

Uses `RFMDataset` with `return_npz_paths=True` to avoid loading frames. Total samples = `dataset_size × num_epochs`.

### Basic Usage

```bash
python scripts/generate_vqa_dataset.py \
    --num_epochs <epochs> \
    --output_path <path> \
    --config_overrides data.train_datasets=[<dataset>]
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_epochs` | Required | Epochs to iterate (e.g., 0.5, 1.0, 10.0) |
| `--output_path` | Required | Output directory |
| `--num_workers` | 4 | DataLoader workers |
| `--batch_size` | 100 | DataLoader batch size |
| `--save_batch_size` | 50000 | Save every N samples (prevents OOM) |
| `--eval_mode` | False | No augmentations (for eval sets) |
| `--seed` | 42 | Random seed |

### Examples

```bash
# Small training set (10% = ~4.3k samples)
python scripts/generate_vqa_dataset.py \
    --num_epochs 0.1 --output_path ./vqa_datasets/train_small \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]

# Full training set (1 epoch = ~43k samples)
python scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 --output_path ./vqa_datasets/train \
    --num_workers 8 \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]

# Large training set (10 epochs = ~430k samples)
python scripts/generate_vqa_dataset.py \
    --num_epochs 10.0 --output_path ./vqa_datasets/train_large \
    --num_workers 8 --save_batch_size 50000 \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]

# Validation set (no augmentations)
python scripts/generate_vqa_dataset.py \
    --num_epochs 0.1 --output_path ./vqa_datasets/val \
    --eval_mode --seed 123 \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_val]

# Custom ratio: 70% preference, 30% progress
python scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 --output_path ./vqa_datasets/custom \
    --config_overrides \
        data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train] \
        data.sample_type_ratio=[0.7,0.3,0.0]
```

---

## Training

### Single GPU

```bash
# Basic
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --eval_dataset_path ./vqa_datasets/val \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --run_name qwen3_vl_4b

# With Unsloth (2-5x faster, recommended)
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --use_unsloth \
    --run_name qwen3_vl_4b

# With LoRA (memory efficient)
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --use_unsloth --lora_rank 16 --lora_alpha 32 \
    --run_name qwen3_vl_4b_lora

# Frozen vision tower (even more memory efficient)
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --use_unsloth --freeze_vision_tower \
    --run_name qwen3_vl_4b_frozen

# With W&B logging
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --use_unsloth \
    --report_to wandb --wandb_project rfm --wandb_entity your-entity \
    --run_name qwen3_vl_4b
```

### Multi-GPU (Accelerate - Recommended)

```bash
# Auto-detect all GPUs
accelerate launch scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --use_unsloth \
    --run_name qwen3_vl_4b_multi

# Specify 4 GPUs explicitly
accelerate launch --num_processes=4 scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --use_unsloth \
    --run_name qwen3_vl_4b_4gpu

# Or use provided script (edit scripts/launch_multi_gpu.sh)
bash scripts/launch_multi_gpu.sh
```

### Multi-GPU (Torchrun)

```bash
# Auto-detect GPUs
torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
    scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --use_unsloth \
    --run_name qwen3_vl_4b_ddp

# Specify 4 GPUs
torchrun --nproc_per_node=4 scripts/train_vqa_sft.py [... args ...]

# Or use provided script (edit scripts/launch_torchrun.sh)
bash scripts/launch_torchrun.sh
```

### Multi-GPU Tips

**Effective Batch Size:** `per_device_batch × grad_accum × num_gpus`

| GPUs | Per-Device | Grad Accum | Effective | LR |
|------|-----------|-----------|----------|-----|
| 1 | 4 | 4 | 16 | 2e-5 |
| 2 | 4 | 2 | 16 | 1e-5 |
| 4 | 2 | 2 | 16 | 5e-6 |
| 8 | 2 | 1 | 16 | 2.5e-6 |

**Rule:** Scale LR down with more GPUs: `LR_multi = LR_single / num_gpus`

---

## Resuming Training

```bash
# Auto-resume from latest checkpoint
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --output_dir ./outputs/qwen3_vl_4b \
    --resume_from_checkpoint True \
    --run_name qwen3_vl_4b_resumed \
    [... same args as initial training ...]

# Resume from specific checkpoint
python scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --output_dir ./outputs/qwen3_vl_4b \
    --resume_from_checkpoint ./outputs/qwen3_vl_4b/checkpoint-1000 \
    --run_name qwen3_vl_4b_from_1000 \
    [... same args ...]

# Works with multi-GPU too
accelerate launch scripts/train_vqa_sft.py \
    --resume_from_checkpoint True \
    [... args ...]
```

Resumes: model weights, optimizer state, LR scheduler, training step, RNG states.

---

## Monitoring

### Metrics

Automatically logged during evaluation:
- `eval_loss`: Cross-entropy loss
- `eval_preference_accuracy`: Preference prediction accuracy
- `eval_progress_mae`: Progress MAE (0-1 scale)
- `eval_progress_rmse`: Progress RMSE

### TensorBoard

```bash
tensorboard --logdir ./outputs
# View at http://localhost:6006
```

### Weights & Biases

Add to training command:
```bash
--report_to wandb --wandb_project rfm --wandb_entity your-entity
```

---

## Key Arguments

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | Required | Training dataset path |
| `--eval_dataset_path` | None | Eval dataset path |
| `--model_name` | Qwen/Qwen3-VL-4B-Instruct | Model to train |
| `--output_dir` | ./outputs/vqa_training | Output directory |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 2 | Gradient accumulation |
| `--num_train_epochs` | 1 | Number of epochs |
| `--learning_rate` | 2e-5 | Learning rate |
| `--lr_scheduler_type` | cosine | LR scheduler |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--save_steps` | 500 | Save every N steps |
| `--eval_steps` | 500 | Eval every N steps |
| `--run_name` | Required | Run name for logging |

### Memory Optimization
| Argument | Description |
|----------|-------------|
| `--use_unsloth` | 2-5x faster training |
| `--quantization` | 4-bit quantization (needs Unsloth) |
| `--freeze_vision_tower` | Only train LLM + projector |
| `--lora_rank 16` | Use LoRA adapters |

### Resumption
| Argument | Description |
|----------|-------------|
| `--resume_from_checkpoint True` | Auto-resume from latest |
| `--resume_from_checkpoint <path>` | Resume from specific checkpoint |

---

## Memory Optimization

| Strategy | Command | Memory Saved | Trade-off |
|----------|---------|--------------|-----------|
| Unsloth | `--use_unsloth` | ~30% | None (faster!) |
| Frozen Vision | `--freeze_vision_tower` | ~40% | Slight perf loss |
| LoRA | `--lora_rank 16` | ~60% | Slower convergence |
| 4-bit Quant | `--quantization` | ~75% | Small accuracy loss |
| Small Batch | `--per_device_train_batch_size 1` | Scales | Slower training |

**Combine strategies:** `--use_unsloth --freeze_vision_tower` saves ~60% memory.

---

## Troubleshooting

**Out of Memory:**
- Reduce `--per_device_train_batch_size` (try 1 or 2)
- Increase `--gradient_accumulation_steps`
- Add `--freeze_vision_tower`
- Add `--lora_rank 16` or `--quantization`

**Slow Training:**
- Add `--use_unsloth` (2-5x speedup)
- Use multi-GPU training
- Increase batch size if memory allows
- Reduce `--eval_steps` frequency

**Poor Performance:**
- Generate more training data (higher `--num_epochs`)
- Train longer (`--num_train_epochs 3`)
- Tune learning rate (try 1e-5 to 5e-5)
- Remove `--freeze_vision_tower` for vision-heavy tasks
- Use full finetuning instead of LoRA

**Multi-GPU: Port in Use:**
```bash
accelerate launch --main_process_port=29501 scripts/train_vqa_sft.py [...]
# or
torchrun --master_port=29501 scripts/train_vqa_sft.py [...]
```

---

## Performance

### Speed (Qwen3-VL-4B, 43k samples, 1 epoch)

| Setup | GPUs | Time | Speedup |
|-------|------|------|---------|
| Basic | 1 | ~12 hrs | 1.0x |
| + Unsloth | 1 | ~5 hrs | 2.4x |
| Multi-GPU | 2 | ~3 hrs | 4.0x |
| Multi + Unsloth | 2 | ~1.5 hrs | 8.0x |
| Multi-GPU | 4 | ~1.5 hrs | 8.0x |
| Multi + Unsloth | 4 | ~45 min | 16.0x |

### Memory (Qwen3-VL-4B per GPU)

| Config | Memory |
|--------|--------|
| Full finetune | ~40 GB |
| + Frozen vision | ~25 GB |
| + LoRA (r=16) | ~20 GB |
| + 4-bit quant | ~15 GB |

---

## Complete Example

```bash
# 1. Generate datasets
python scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 --output_path ./vqa_datasets/train \
    --num_workers 8 \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]

python scripts/generate_vqa_dataset.py \
    --num_epochs 0.1 --output_path ./vqa_datasets/val \
    --eval_mode --seed 123 --num_workers 8 \
    --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_val]

# 2. Train on 4 GPUs
accelerate launch scripts/train_vqa_sft.py \
    --dataset_path ./vqa_datasets/train \
    --eval_dataset_path ./vqa_datasets/val \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_unsloth \
    --eval_strategy steps --eval_steps 500 \
    --save_steps 500 \
    --report_to wandb --wandb_project rfm \
    --run_name qwen3_vl_4b_roboreward

# 3. Monitor
tensorboard --logdir ./outputs/qwen3_vl_4b

# 4. Resume if interrupted
accelerate launch scripts/train_vqa_sft.py \
    --resume_from_checkpoint True \
    [... same args ...]
```

---

## Best Practices

**Dataset Generation:**
- Use `--num_workers 8` for speed
- Use `--eval_mode` for validation sets
- Use `--save_batch_size 50000` for large datasets

**Training:**
- Always use `--use_unsloth` for faster training
- Use cosine LR schedule (default)
- Evaluate every 500-1000 steps
- Scale LR down with more GPUs
- Use W&B for tracking experiments

**Multi-GPU:**
- Prefer Accelerate over Torchrun
- Keep per-device batch size 2-4
- Monitor GPU utilization with `nvidia-smi`
- Ensure .npz files on fast storage (SSD)

---

For more details, see individual script docstrings or [VQA_TRAINING_SUMMARY.md](VQA_TRAINING_SUMMARY.md).
