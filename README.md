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

## Configuration

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
