# Reward Foundation Model (RFM)

PyTorch implementation of a Reward Foundation Model for robotic learning.

## Quick Setup

```bash
git clone https://github.com/aliang8/reward_fm.git
cd reward_fm
make install
```

## Dataset Setup

Set your dataset path (optional):
```bash
export RFM_DATASET_PATH=/path/to/your/rfm_dataset
```

Download datasets:
```bash
# Just LIBERO (recommended for testing)
make dataset-libero

# Or download all datasets (~600GB+)
make dataset-all
```

## Usage

```bash
# Training
make train

# Evaluation
make eval

# Check status
make status
```

### Evaluation via HTTP server
You can run evaluations through a lightweight HTTP server that hosts the model and returns metrics.

Start the server (optionally override YAML fields with --set):
```bash
uv run evals/qwen_server.py \
  --config_path=rfm/configs/config.yaml \
  --host=0.0.0.0 --port=8000 \
  --set 'evaluation.model_path="aliangdw/rfm_v1"'
```

Run the external client to send video batches and receive metrics:
```bash
uv run python evals/run_model_eval.py \
  --config_path=rfm/configs/config.yaml \
  --server_url=http://localhost:8000 \
  --batch_size=15 \
  --iterate_all_preferences
```

Optionally, trigger full internal evaluation (same flow as train.py evaluate):
```bash
curl -X POST http://localhost:8000/evaluate_internal \
  -H 'Content-Type: application/json' \
  -d '{"eval_subset_size": 100}'
```

Notes:
- Set `RFM_DATASET_PATH` to the directory holding your downloaded datasets so the server/client can resolve video paths.
- The external endpoint `/evaluate_batch` accepts pre-batched, base64-encoded videos and returns per-batch metrics.
- The internal endpoint `/evaluate_internal` reuses the trainer evaluation pipeline to compute a full eval in one call.

## Development

Training configs are in `rfm/configs/`. Modify `config.yaml` for your experiments.

## Manual Installation

If you prefer manual setup:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download LIBERO only
uv run huggingface-cli download abraranwar/libero_rfm \
  --repo-type dataset \
  --local-dir ./rfm_dataset/libero_rfm

# Run training
uv run accelerate launch --config_file rfm/configs/fsdp.yaml train.py --config_path=rfm/configs/config.yaml
```
