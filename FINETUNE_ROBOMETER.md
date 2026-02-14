# Fine-tuning Robometer on Your Own Data

Preprocess a dataset, LoRA fine-tune from Robometer-4B, upload to the Hub, run inference. Example: [MINT-SJTU/RoboFAC-dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset).

---

## 1. Preprocessing (RoboFAC)

Training needs a **preprocessed** cache from a HuggingFace dataset in RBM format. RoboFAC is folder-based; convert first, then preprocess.

1. **Download** (into `ROBOMETER_DATASET_PATH`):
   ```bash
   export ROBOMETER_DATASET_PATH=/path/to/your/robometer_dataset
   huggingface-cli download MINT-SJTU/RoboFAC-dataset --local-dir $ROBOMETER_DATASET_PATH/RoboFAC-dataset
   ```

2. **Convert and push to Hub** (required for recommended flow):
   ```bash
   export HF_TOKEN=your_token_here

   uv run python -m dataset_upload.generate_hf_dataset \
     --config_path dataset_upload/configs/data_gen_configs/robofac.yaml \
     --dataset.dataset_path=$ROBOMETER_DATASET_PATH/RoboFAC-dataset \
     --hub.push_to_hub=true \
     --hub.hub_repo_id=robofac_rbm
   ```

3. **If you pushed:** download converted repo and set preprocess config:
   ```bash
   huggingface-cli download YOUR_USERNAME/robofac_rbm --local-dir $ROBOMETER_DATASET_PATH/robofac_rbm
   ```
   In `preprocess_finetune.yaml`: `train_datasets: ["YOUR_USERNAME/robofac_rbm"]`, `train_subsets: [["robofac"]]`.

4. **Preprocess:**
   ```bash
   export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets

   uv run python -m robometer.data.scripts.preprocess_datasets \
     --config robometer/configs/preprocess_finetune.yaml \
     --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
   ```
   Use the same path for training.

**Other datasets:** RBM-style Hub datasets: set `train_datasets`/`train_subsets` and `ROBOMETER_DATASET_PATH` in the preprocess config, then run step 4. Raw data: add a loader (see [CustomDataset.md](dataset_upload/dataset_guides/CustomDataset.md)).

---

## 2. LoRA fine-tuning

Use PEFT (LoRA) and `resume_from_checkpoint` from a Qwen3-4B–based RBM checkpoint.

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/your/processed_datasets

uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[YOUR_USERNAME/robofac_rbm/robofac] \
  data.eval_datasets=[YOUR_USERNAME/robofac_rbm/robofac] \
  training.load_from_checkpoint=aliangdw/Robometer-4B \
  training.max_steps=500 \
  training.output_dir=./logs \
  training.exp_name=robometer4b_lora_robofac
```

Multi-GPU: `uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py ...` (same overrides). Tune `max_steps`, `learning_rate`, PEFT in `robometer/configs/config.yaml`.

### Full fine-tuning (no PEFT)

Load the same checkpoint but train the full model (no LoRA). Uses more memory; lower `per_device_train_batch_size` or gradient accumulation if needed.

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/your/processed_datasets

uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[YOUR_USERNAME/robofac_rbm/robofac] \
  data.eval_datasets=[YOUR_USERNAME/robofac_rbm/robofac] \
  training.load_from_checkpoint=aliangdw/Robometer-4B \
  training.max_steps=500 \
  training.output_dir=./logs \
  training.exp_name=robometer4b_full_robofac
```

---

## 3. Upload to Hub

```bash
uv run python robometer/utils/upload_to_hub.py \
  --model_dir ./logs/robometer4b_lora_robofac/checkpoint-500 \
  --hub_model_id YOUR_USERNAME/robometer-4b-lora-robofac \
  --base_model "Qwen/Qwen3-VL-4B-Instruct" \
  --commit_message "LoRA fine-tune on RoboFAC"
```

Or enable `logging.save_best.upload_to_hub: true` in config for upload during training.

---

## 4. Inference

```bash
uv run python scripts/example_inference_local.py \
  --model-path YOUR_USERNAME/robometer-4b-lora-robofac \
  --video /path/to/video.mp4 \
  --task "Insert the cylinder"
```

Server: `uv run python robometer/evals/eval_server.py ... model_path=YOUR_USERNAME/robometer-4b-lora-robofac`. Eval: `run_baseline_eval.py` with `reward_model=rbm`, `model_path=...` (see [README](README.md)).
