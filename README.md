# Reward Foundation Model (RFM)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aliang8/reward_fm/ci.yml?branch=main&style=for-the-badge)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)
![License](https://img.shields.io/github/license/aliang8/reward_fm?style=for-the-badge)

This repository contains the official PyTorch implementation for "Reward Foundation Model" (RFM). 

## Package Structure

The project is organized as follows:

```
reward_fm/
├── rfm/                    # Main package directory
│   ├── data/              # Data processing and dataset utilities
│   ├── configs/           # Configuration files
│   └── models/            # Model definitions
├── rfm_visualizer/        # Gradio web interface for dataset visualization
├── train.py              # Main training script
├── setup.py              # Package installation script
└── pyproject.toml        # Project configuration
```

## Setup and Installation

This project can be installed as a Python package called `rfm`. We provide multiple installation methods:

### Method 1: Install as a Package (DEPRECATED, DON"T use)

Install the RFM package directly:

```bash
# Clone the repository
git clone https://github.com/aliang8/reward_fm.git
cd reward_fm

# Install the package in development mode
pip install -e .

# Or use the installation script
python install.py
```

After installation, you can use the package:

```python
import rfm

# Access the visualizer
from rfm_visualizer.app import demo
demo.launch()

# Or use the console scripts
# rfm-visualizer  # Launch the visualizer
# rfm-train       # Run training
```

### Method 2: Using uv (Development)

This project uses `uv` for fast and reliable dependency management. We recommend it over `pip` and `venv` for a much better developer experience.

### Prerequisites

*   Git
*   Python 3.10+
*   NVIDIA Drivers (for GPU support)

### Step-by-Step Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/aliang8/reward_fm.git
    cd reward_fm
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, you can install it with this command:
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    (See the [official uv installation guide](https://github.com/astral-sh/uv#installation) for other systems).

3.  **Create and Activate a Virtual Environment:**
    Install with:
    ```
    uv sync
    ```

    You can separately activate your environment with
    ```
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    This project uses `pyproject.toml` for dependency management. Use the `uv sync` command to install all required packages.

    ```bash
    # Install dependencies from pyproject.toml
    uv sync
    
    # Optional: Install with development dependencies
    uv sync --extra dev
    
    # Optional: Install with quantization support (Linux/Windows only)
    uv sync --extra quantization
    ```

    **Note for macOS users:** The `bitsandbytes` package for model quantization is not available on macOS ARM. It's included as an optional dependency that can be installed on compatible platforms.

5.  **Verify Installation:**
    ```bash
    # Check that the environment is properly set up
    uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    uv run python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    ```
### Next: Dataset Setup
We now download the dataset to the local `./rfm_dataset` directory (by default).
For space reasons, you should symlink `~/.cache/huggingface/datasets` to some other location with ample space first, as that's where the dataset is downloaded to by default before being symlinked to `./rfm_dataset`.

```bash
# Download the dataset
./setup.sh
```

Add to your `.bashrc` the following export:

```bash
export RFM_DATASET_PATH=/path/to/your/rfm_dataset
```



### Troubleshooting

**Legacy requirements.lock.txt:** If you encounter issues with the old `requirements.lock.txt` file, ignore it and use `uv sync` instead, which reads dependencies from `pyproject.toml`.

**Dependency Management:** `uv` automatically generates a `uv.lock` file for reproducible builds. This file is similar to `requirements.lock.txt` but properly handles platform-specific dependencies and version resolution.

## Dataset Generation

RFM supports multiple robotic datasets with automatic video processing:

### Supported Datasets
- **🚀 AgiBotWorld**: Large-scale dataset with streaming support and automatic video processing (256x256, frame interpolation)
- **🔧 LIBERO**: Simulation dataset with local HDF5 support
- **⚙️ Custom**: Template for adding new datasets

### Quick Examples
```bash
# AgiBotWorld (streaming, ~600GB dataset)
uv run python rfm/data/generate_hf_dataset.py --config_path=rfm/configs/data_gen_configs/agibot_world.yaml

# LIBERO (local files)
uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen.yaml \
    --dataset.dataset_path=LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90

# Custom parameters
uv run python rfm/data/generate_hf_dataset.py \
    --dataset.dataset_name=agibotworld \
    --output.max_frames=16 \
    --output.max_trajectories=100
```

📖 **Detailed Guide**: [rfm/data/README_ADDING_DATASETS.md](rfm/data/README_ADDING_DATASETS.md)

## Training and Evaluation
```bash
# Training
uv run accelerate launch --config_file rfm/configs/fsdp.yaml train.py --config_path=rfm/configs/config.yaml

# Evaluation
uv run accelerate launch --config_file rfm/configs/fsdp.yaml train.py --mode=evaluate
```

### Evaluation via HTTP server
You can run evaluations through a lightweight HTTP server that hosts the model and returns metrics.

Start the server (optionally override YAML fields with --set):
```bash
uv run evals/qwen_server.py \
  --config_path=rfm/configs/config.yaml \
  --host=0.0.0.0 --port=8000 \
  --set 'evaluation.model_path="aliangdw/rfm_v0"'
```

Run the external client to send video batches and receive metrics:
```bash
uv run python evals/run_model_eval.py \
  --config_path=rfm/configs/config.yaml \
  --server_url=http://localhost:8000 \
  --num_batches=50 \
  --batch_size=12
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

Please ensure your code adheres to the style guidelines by running the linter:
```bash
# Check code style (requires dev dependencies)
uv run ruff check .

# Format code
uv run ruff format .
```

## License

This project is licensed under the [MIT License](LICENSE).