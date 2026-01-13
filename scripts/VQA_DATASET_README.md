# VQA Dataset Generation & Training Scripts

This directory contains two scripts for generating VQA-style datasets from RFM trajectories and training Qwen3-VL models on them.

## Overview

The scripts enable a two-stage workflow:
1. **Dataset Generation**: Pre-generate a static VQA dataset with prompts, answers, and references to .npz video files
2. **Training**: Train Qwen3-VL models using standard HuggingFace Trainer on the generated dataset

This approach offers several advantages over the original dynamic sampling:
- **Reproducibility**: Static datasets enable exact reproduction of results
- **Debugging**: Inspect generated samples before training
- **Flexibility**: Use the dataset with any training framework
- **Simplicity**: Standard HF code without custom complexity

## Files

- `generate_vqa_dataset.py`: Generate VQA dataset from RFM trajectories
- `train_vqa_sft.py`: Train Qwen3-VL on the generated dataset
- `test_vqa_collator.py`: Test script to verify collator functionality

## 1. Dataset Generation

### Usage

```bash
python scripts/generate_vqa_dataset.py \
    --num_samples 10000 \
    --output_path /path/to/output/dataset \
    --seed 42 \
    --config_overrides data.max_frames=16 data.sample_type_ratio=[0.7,0.3,0.0]
```

### Arguments

- `--num_samples`: Number of samples to generate (default: 10000)
- `--output_path`: Path to save the generated HuggingFace dataset (required)
- `--seed`: Random seed for reproducibility (default: 42)
- `--config_name`: Hydra config to use (default: "config")
- `--config_overrides`: Config overrides in key=value format (optional)

### Configuration

The script uses the same Hydra configuration as `train.py`:
- Config file: `rfm/configs/config.yaml`
- Modify `data.sample_type_ratio` to control preference vs progress ratio
  - `[1, 0, 0]`: Only preference samples
  - `[0, 1, 0]`: Only progress samples
  - `[0.7, 0.3, 0]`: 70% preference, 30% progress
- Modify `data.max_frames` to control frames per video
- Use `--config_overrides` to override any config value

### Output

The script generates:
- **HuggingFace Dataset**: Saved at `output_path/`
- **Generation Config**: Saved at `output_path/generation_config.json`

### Dataset Structure

#### Preference Samples
```json
{
  "sample_type": "preference",
  "prompt": "Given these two robot videos, which one makes the most progress towards solving the task, Video 1 or 2? Format your answer as: ANS: 1/2\n\nTask: push button",
  "answer": "1",
  "first_npz_path": "path/to/first_video.npz",
  "second_npz_path": "path/to/second_video.npz",
  "first_frame_indices": [0, 5, 10, 15],
  "second_frame_indices": [0, 3, 6, 9],
  "first_frames_shape": [16, 224, 224, 3],
  "second_frames_shape": [16, 224, 224, 3],
  "task": "push button",
  "data_source": "metaworld_train",
  "data_gen_strategy": "rewind",
  "resample_attempts": 1,
  "chosen_is_first": true
}
```

#### Progress Samples
```json
{
  "sample_type": "progress",
  "prompt": "Given the task, assign an integer-valued progress score from 0 to 100 for the robot in the video...",
  "answer": "75",
  "npz_path": "path/to/video.npz",
  "frame_indices": [0, 5, 10, 15],
  "frames_shape": [16, 224, 224, 3],
  "target_progress": [0.0, 0.25, 0.5, 0.75],
  "task": "open drawer",
  "data_source": "metaworld_train",
  "data_gen_strategy": "forward_progress"
}
```

### Implementation Details

- **No Frame Loading**: The generation script does NOT load actual video frames into memory
- **NPZ References**: Stores relative paths to .npz files and frame indices
- **Lazy Loading**: Frames are loaded on-the-fly during training
- **Memory Efficient**: Can generate millions of samples without memory issues

## 2. Training

### Usage

```bash
python scripts/train_vqa_sft.py \
    --dataset_path /path/to/generated/dataset \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/vqa_training \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

### Arguments

**Dataset:**
- `--dataset_path`: Path to the generated HuggingFace dataset (required)
- `--eval_dataset_path`: Path to evaluation dataset (optional)

**Model:**
- `--model_name`: Model name or path (default: "Qwen/Qwen3-VL-4B-Instruct")
  - Supports: Qwen3-VL-4B, Qwen3-VL-8B, Qwen2.5-VL-3B, etc.
- `--use_multi_image`: Use multi-image mode instead of video mode (flag)

**Training:**
- `--output_dir`: Output directory for checkpoints (default: "./outputs/vqa_training")
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--per_device_eval_batch_size`: Eval batch size per device (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--num_train_epochs`: Number of epochs (default: 3)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--warmup_ratio`: Warmup ratio (default: 0.1)
- `--max_grad_norm`: Gradient clipping (default: 1.0)
- `--weight_decay`: Weight decay (default: 0.01)
- `--bf16`: Use bfloat16 (default: True)
- `--gradient_checkpointing`: Enable gradient checkpointing (default: True)

**Saving:**
- `--save_strategy`: Save strategy: "no", "steps", "epoch" (default: "steps")
- `--save_steps`: Save every N steps (default: 500)
- `--logging_steps`: Log every N steps (default: 10)

**Evaluation:**
- `--eval_strategy`: Eval strategy: "no", "steps", "epoch" (default: "steps")
- `--eval_steps`: Evaluate every N steps (default: 500)

### Training Features

**Custom VQA Collator:**
- Loads frames from .npz files on-the-fly
- Subsamples frames based on stored indices
- Converts to PIL images
- Formats as Qwen3-VL conversation
- Tokenizes with processor
- Masks prompt tokens (only trains on answer)

**Supported Models:**
- Qwen3-VL-4B-Instruct
- Qwen3-VL-8B-Instruct
- Qwen2.5-VL-3B-Instruct
- Other Qwen VL models

**Training Mode:**
- Standard supervised fine-tuning (SFT)
- Trains only on answer tokens (prompt is masked)
- Uses standard HuggingFace Trainer
- No custom trainers or models required

## 3. Testing

### Test Collator

Verify the collator works correctly:

```bash
python scripts/test_vqa_collator.py
```

This script:
1. Loads a small dataset
2. Initializes the collator
3. Processes a batch
4. Verifies frame loading and tokenization
5. Checks label masking

## Example Workflow

### 1. Generate Dataset

```bash
# Generate 10k samples with 70% preference, 30% progress
python scripts/generate_vqa_dataset.py \
    --num_samples 10000 \
    --output_path /data/vqa_dataset_10k \
    --seed 42 \
    --config_overrides data.sample_type_ratio=[0.7,0.3,0.0]
```

### 2. Train Model

```bash
# Train Qwen3-VL-4B on the generated dataset
python scripts/train_vqa_sft.py \
    --dataset_path /data/vqa_dataset_10k \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs/qwen3_vl_4b_vqa \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing
```

### 3. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir ./outputs/qwen3_vl_4b_vqa
```

## Implementation Notes

### Dataset Generation
- Works directly with trajectory dictionaries from HF datasets
- Extracts npz paths and frame indices WITHOUT loading frames
- Computes target progress using same logic as samplers
- Randomly assigns which trajectory is "Video 1" or "Video 2" for preference samples

### Training Collator
- Loads frames from npz files using `load_frames_from_npz()`
- Subsamples frames based on stored `frame_indices`
- Converts to PIL images using `convert_frames_to_pil()`
- Formats as Qwen conversation with user/assistant roles
- Uses `qwen_vl_utils.process_vision_info()` for vision processing
- Handles both Qwen2.5-VL and Qwen3-VL formats

### Label Masking
- Masks all tokens before "ANS:" response prefix
- Only trains on the answer portion (e.g., "1", "2", or "75")
- Uses IGNORE_INDEX = -100 for masked tokens
- Typically masks ~99.7% of tokens (only 1-2 answer tokens remain)

## Troubleshooting

### Dataset Generation Issues

**Error: "Could not load 'base_config'"**
- The script now registers configs automatically
- If you still see this, ensure `rfm/configs/experiment_configs.py` is accessible

**Error: "Frames are loaded as numpy array"**
- This should not happen with the fixed script
- The script now works directly with trajectory dicts without calling samplers

### Training Issues

**Error: "not enough values to unpack (expected 3, got 2)"**
- Fixed: The collator now handles both Qwen2.5 (2 values) and Qwen3 (3 values)

**Out of Memory**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `--gradient_checkpointing` (default: enabled)
- Use a smaller model (e.g., Qwen2.5-VL-3B instead of Qwen3-VL-8B)

**Slow Training**
- Reduce `--per_device_eval_batch_size`
- Increase `--eval_steps` to evaluate less frequently
- Set `--dataloader_num_workers` appropriately (default: 4)
- Ensure `.npz` files are on fast storage (SSD, not network drive)

## Advantages Over Dynamic Sampling

1. **Reproducibility**: Static dataset = exact same samples every time
2. **Debugging**: Inspect samples before training
3. **Flexibility**: Use with any framework (PyTorch Lightning, Composer, etc.)
4. **Simplicity**: Standard HF code, no custom trainers
5. **Efficiency**: Pre-computed samples, no resampling overhead
6. **Distribution Control**: Exact control over sample type ratios
7. **Data Inspection**: Analyze dataset statistics before training

## Disadvantages

1. **Storage**: Requires disk space for dataset metadata (npz files already exist)
2. **Fixed Data**: Can't change sampling strategies mid-training
3. **Upfront Cost**: Must generate dataset before training

## Future Enhancements

Potential improvements:
- [ ] Support for similarity samples
- [ ] Multi-strategy sampling (different strategies per sample)
- [ ] Data augmentation (random frame dropping, color jitter)
- [ ] Inference script for evaluating trained models
- [ ] Automatic dataset splitting (train/val/test)
- [ ] Dataset statistics and visualization tools
