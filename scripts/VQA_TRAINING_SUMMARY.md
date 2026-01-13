# VQA Training with Automatic Evaluation

## Key Changes

### 1. Evaluation During Training ✅

The training script now automatically evaluates VQA metrics during training when you provide `--eval_dataset_path`:

**Metrics Computed:**
- `eval_loss` - Standard cross-entropy loss
- `eval_preference_accuracy` - Preference prediction accuracy (% correct)
- `eval_progress_mae` - Progress MAE (Mean Absolute Error)
- `eval_progress_rmse` - Progress RMSE (Root Mean Squared Error)

**How it works:**
- Custom `VQAEvaluationCallback` runs after each evaluation
- Generates answers on eval set using greedy decoding
- Extracts answers from "ANS: X" format
- Computes metrics and logs to TensorBoard

### 2. Training Command

```bash
python scripts/train_vqa_sft.py \
    --dataset_path /data/vqa_train \
    --eval_dataset_path /data/vqa_test \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs \
    --eval_strategy steps \
    --eval_steps 500 \
    --use_unsloth
```

### 3. Example Output

During training, you'll see:

```
================================================================================
Running VQA Evaluation (Generation-based)
================================================================================
Preference Accuracy: 0.9214 (647/702)
Progress MAE: 6.45
Progress RMSE: 9.23
================================================================================
{'loss': 0.1234, 'learning_rate': 1.8e-05, 'epoch': 1.5}
{'eval_loss': 0.0987, 'eval_preference_accuracy': 0.9214, 'eval_progress_mae': 6.45, 'eval_progress_rmse': 9.23, 'epoch': 1.5}
```

### 4. Monitoring with TensorBoard

```bash
tensorboard --logdir ./outputs
```

You'll see plots for:
- Training loss
- Eval loss
- Preference accuracy (over time)
- Progress MAE (over time)
- Progress RMSE (over time)

### 5. Optional: Post-Training Analysis

For detailed per-sample analysis, you can still use:

```bash
python scripts/evaluate_vqa.py \
    --model_path ./outputs/final \
    --dataset_path /data/vqa_test \
    --output_path ./eval_results.json \
    --save_predictions
```

This gives you individual predictions for error analysis.

## Complete Example

```bash
# 1. Generate datasets
python scripts/generate_vqa_dataset.py --num_samples 10000 --output_path /data/train --seed 42
python scripts/generate_vqa_dataset.py --num_samples 1000 --output_path /data/test --seed 123

# 2. Train with automatic evaluation
python scripts/train_vqa_sft.py \
    --dataset_path /data/train \
    --eval_dataset_path /data/test \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./outputs \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --use_unsloth \
    --bf16

# 3. Monitor training
tensorboard --logdir ./outputs

# 4. (Optional) Detailed analysis
python scripts/evaluate_vqa.py \
    --model_path ./outputs/final \
    --dataset_path /data/test \
    --output_path ./eval_results.json \
    --save_predictions
```

## Benefits

1. **Real-time metrics**: See VQA performance during training, not just loss
2. **Early stopping**: Monitor accuracy/MAE to stop when performance plateaus
3. **Hyperparameter tuning**: Compare different runs by VQA metrics, not just loss
4. **No extra work**: Automatic when you provide eval dataset
5. **TensorBoard integration**: All metrics logged and visualized

## Notes

- Evaluation uses greedy decoding (do_sample=False) for deterministic results
- Batch size for evaluation can be set via `--per_device_eval_batch_size`
- Evaluation runs every `--eval_steps` during training
- Works with both standard training and unsloth
- No need for separate evaluation script unless you want detailed per-sample analysis
