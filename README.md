# Robometer: Scalable Reward Modeling with Progress + Preference

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org/)
[![GitHub](https://img.shields.io/badge/GitHub-reward__fm-181717?logo=github)](https://github.com/aliang8/reward_fm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-RBM--1M-FFD21E?logo=huggingface)](https://huggingface.co/datasets/)
[![Visualizer](https://img.shields.io/badge/🖼️%20Visualizer-HuggingFace%20Space-FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/visualizer)
[![Eval UI](https://img.shields.io/badge/📊%20Eval%20UI-HuggingFace%20Space-FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/rewardeval_ui)

**arXiv (Coming Soon)** | **Project Homepage** | **🤗 Model** | **🤗 Dataset (RBM-1M)** | **🖼️ [Visualizer](https://huggingface.co/spaces/rewardfm/visualizer)** | **📊 [Eval UI](https://huggingface.co/spaces/rewardfm/rewardeval_ui)** | **Benchmark**

<p align="center">
  <img src="assets/robometer.jpg" alt="Robometer" width="100%"/>
</p>

---

## Abstract

General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce **Robometer**, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate **RBM-1M**, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications.

---

## 🗞️ News

- **Baseline evals**: Run reward alignment, policy ranking, and confusion matrix evals for RFM, ReWiND, GVL, VLAC, RoboReward, and **Robometer** (Robo-Dopamine GRM).

---

## 🤖 Overview

**Robometer** provides:

- **Training**: Progress and preference reward modeling over trajectory videos and task descriptions.
- **Baseline evaluations**: Compare against GVL, VLAC, RoboReward, and **Robometer** (Robo-Dopamine GRM) on reward alignment, policy ranking, and confusion matrix metrics.

---

## 📦 Package structure

```
reward_fm/
├── robometer/              # Main package
│   ├── data/               # Datasets and preprocessing
│   ├── configs/            # Hydra and experiment configs
│   ├── models/             # Model definitions
│   └── evals/              # Baseline evals (GVL, VLAC, Robometer, etc.)
├── eval_commands/          # Shell scripts for baseline evals
├── train.py                # Training entrypoint
└── pyproject.toml          # Dependencies (uv)
```

---

## 🛠️ Setup

### Prerequisites

- Git, Python 3.10+
- NVIDIA drivers (GPU)
- [uv](https://github.com/astral-sh/uv#installation) (recommended)

### Install (main env)

```bash
git clone https://github.com/aliang8/reward_fm.git
cd reward_fm

# Create venv and install
uv sync
```

### Dataset setup

```bash
hf auth
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
./scripts/download_processed_datasets.sh
./scripts/untar_processed_datasets.sh
```

For raw download and preprocessing, see [📥 Download raw datasets](#-download-raw-datasets-optional) below.

---

### 🔍 Robometer evaluation

Same interface as other baselines: `reward_model=robodopamine`, plus `policy_ranking` or `confusion_matrix` eval types. Example commands are in `eval_commands/reward_alignment.sh`, `eval_commands/policy_ranking.sh`, and `eval_commands/confusion_matrix.sh`.  
Detailed baseline eval docs: [robometer/evals/README.md](robometer/evals/README.md).

---

## 🏋️ Training and evaluation

### Training

```bash
uv run accelerate launch --config_file robometer/configs/fsdp.yaml train.py --config_path=robometer/configs/config.yaml
```

### Evaluation (train script)

```bash
uv run accelerate launch --config_file robometer/configs/fsdp.yaml train.py --mode=evaluate
```

### Baseline evaluation (all models)

```bash
# RFM / ReWiND
uv run python robometer/evals/run_baseline_eval.py reward_model=rfm model_path=... custom_eval.eval_types=[reward_alignment] ...

# GVL, VLAC, RoboReward: see robometer/evals/README.md and eval_commands/*.sh
# Robometer: use .venv-robodopamine/bin/python as in [Robometer inference](#-robometer-inference-reward-alignment) above
```

### Evaluation via HTTP server

```bash
# Start server
uv run evals/eval_server.py --config_path=robometer/configs/config.yaml --host=0.0.0.0 --port=8000

# Client
uv run python evals/run_model_eval.py --config_path=robometer/configs/config.yaml --server_url=http://localhost:8000 --batch_size=15 --num-batches=-1
```

---

## 📊 Dataset generation

Supported: **AgiBotWorld** (streaming), **LIBERO** (HDF5), and custom configs.

```bash
# AgiBotWorld
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/agibot_world.yaml

# LIBERO
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen.yaml \
  --dataset.dataset_path=LIBERO/libero/datasets/libero_90 --dataset.dataset_name=libero_90
```

See dataset_upload README and dataset_guides for adding datasets.

---

## 📥 Download raw datasets (optional)

If you prefer not to use the processed datasets:

```bash
export ROBOMETER_DATASET_PATH=/path/to/your/robometer_dataset
./scripts/download_data.sh
# Optional: RFM_DOWNLOAD_METHOD=git ./scripts/download_data.sh

# Preprocess
uv run python -m robometer.data.scripts.preprocess_datasets --config robometer/configs/preprocess.yaml
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
```

---

## 🔧 Development

```bash
uv run ruff check .
uv run ruff format .
```

---

## 📑 License

This project is licensed under the [MIT License](LICENSE).
