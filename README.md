# Robometer: Scaling General-Purpose Robotic
Reward Models via Trajectory Comparisons

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

## 📦 Package structure

```
robometer/
├── robometer/              # Main package
│   ├── data/               # Datasets and preprocessing
│   ├── configs/            # Hydra and experiment configs
│   ├── models/             # Model definitions
│   └── evals/              # Baseline evals (GVL, VLAC, Robodopamine, etc.)
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
git clone https://github.com/aliang8/robometer.git
cd robometer

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

Run RBM baselines with `reward_model=rbm` and override `model_path` and `custom_eval.*` as needed. Example commands (see `eval_commands/*.sh` for ReWIND, Robo-Dopamine, VLAC, RoboReward):

**Reward alignment**

```bash
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=4 \
    model_config.batch_size=32
```

**Policy ranking**

```bash
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=32
```

**Confusion matrix**

```bash
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8 \
    model_config.batch_size=32
```

Detailed baseline eval docs: [robometer/evals/README.md](robometer/evals/README.md).

---

## 🏋️ Training

### Training

```bash
uv run accelerate launch --config_file robometer/configs/fsdp.yaml train.py --config_path=robometer/configs/config.yaml
```

### Baseline evaluation (all models)

- **RBM:** use the [reward alignment](#-robometer-evaluation), [policy ranking](#-robometer-evaluation), or [confusion matrix](#-robometer-evaluation) commands above; set `model_path` to your checkpoint.
- **ReWIND, Robo-Dopamine, VLAC, RoboReward:** see [robometer/evals/README.md](robometer/evals/README.md) and `eval_commands/reward_alignment.sh`, `eval_commands/policy_ranking.sh`, `eval_commands/confusion_matrix.sh`. For Robo-Dopamine use `.venv-robodopamine/bin/python` (vLLM) instead of `uv run`.

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

## 📑 License

This project is licensed under the [MIT License](LICENSE).
