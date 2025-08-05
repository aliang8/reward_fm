# Reward Foundation Model (RFM)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aliang8/reward_fm/ci.yml?branch=main&style=for-the-badge)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)
![License](https://img.shields.io/github/license/aliang8/reward_fm?style=for-the-badge)

This repository contains the official PyTorch implementation for "Reward Foundation Model" (RFM). 

## Setup and Installation

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
    `uv` has a built-in environment manager that is much faster than `venv`.
    ```bash
    # Create a virtual environment in the .venv directory
    uv venv

    # Activate the environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
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
uv run python data/generate_hf_dataset.py --config_path=configs/data_gen_configs/agibot_world.yaml

# LIBERO (local files)
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen.yaml \
    --dataset.dataset_path=LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90

# Custom parameters
uv run python data/generate_hf_dataset.py \
    --dataset.dataset_name=agibotworld \
    --output.max_frames=16 \
    --output.max_trajectories=100
```

📖 **Detailed Guide**: [data/README_ADDING_DATASETS.md](data/README_ADDING_DATASETS.md)

## Training and Evaluation
```bash
# Training
uv run accelerate launch --config_file configs/fsdp.yaml train.py --config_path=configs/config.yaml

# Evaluation
uv run accelerate launch --config_file configs/fsdp.yaml train.py --mode=evaluate
```

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