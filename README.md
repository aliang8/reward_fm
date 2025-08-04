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
    This project includes a `requirements.lock.txt` file for fully reproducible installations. Use the `uv pip sync` command to install the exact versions of all required packages.

    ```bash
    # Sync the virtual environment with the locked dependencies
    uv pip sync -r requirements.lock.txt

## Dataset Generation
```bash
python data/generate_hf_dataset.py \
    --config_path=configs/data_gen.yaml \
    --dataset.dataset_path=LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90
```

## Training and Evaluation
```bash
# training
accelerate launch --config_file configs/fsdp.yaml train.py --config_path=configs/config.yaml

# eval
accelerate launch --config_file configs/fsdp.yaml train.py --mode=evaluate

```

Please ensure your code adheres to the style guidelines by running the linter:
```bash
ruff check .
```

## License

This project is licensed under the [MIT License](LICENSE).