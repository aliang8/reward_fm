#!/usr/bin/env python3
"""
Evaluation configuration for RFM.
This file contains evaluation configuration classes:
- EvalServerConfig: For evaluation server runs (eval_server.py)
- EvalOnlyConfig: For standalone evaluation runs (run_eval_only.py)
"""

from dataclasses import dataclass, field
from typing import Optional

from rfm.configs.experiment_configs import CustomEvaluationConfig
from rfm.configs.experiment_configs import DataConfig


@dataclass
class EvalServerConfig:
    """Configuration for evaluation server runs (eval_server.py)."""

    # Model path to load training config from
    model_path: str = field(
        default="./rfm_model_output/checkpoint-1000",
        metadata={"help": "Path to the trained model checkpoint (will load training_config.yaml from here)"},
    )

    # GPU Pool settings
    num_gpus: int = field(default=1, metadata={"help": "Number of GPUs to use (None for all available)"})
    max_workers: int = field(
        default=None, metadata={"help": "Max worker threads (None for auto, typically same as num_gpus)"}
    )

    # Evaluation parameters
    eval_subset_size: int = field(default=1000, metadata={"help": "Number of samples to evaluate"})
    batch_size: int = field(default=4, metadata={"help": "Batch size for evaluation"})
    num_batches: int = field(default=-1, metadata={"help": "Number of batches to evaluate (-1 for full dataset)"})
    server_url: str = field(default="0.0.0.0", metadata={"help": "Evaluation server URL"})
    server_port: int = field(default=8000, metadata={"help": "Evaluation server port"})
    log_dir: str = field(default="./eval_logs", metadata={"help": "Directory to save evaluation results"})

    custom_eval: CustomEvaluationConfig = field(
        default_factory=CustomEvaluationConfig,
        metadata={"help": "Custom evaluation configuration (reused from experiment_configs)"},
    )

    data: DataConfig = field(
        default_factory=DataConfig, metadata={"help": "Data configuration (reused from experiment_configs)"}
    )


# For backwards compatibility
EvaluationConfig = EvalServerConfig


@dataclass
class EvalOnlyConfig:
    """Configuration for standalone evaluation runs (run_eval_only.py)."""

    # Model path (HuggingFace model ID or local checkpoint path)
    model_path: str = field(
        default="",
        metadata={"help": "HuggingFace model ID (e.g., 'aliangdw/rfm_model') or local checkpoint path"},
    )

    # Optional experiment config override (pyrallis reserved "config_path", so using exp_config_path)
    exp_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to experiment config file (if not provided, will try to load from HuggingFace repo or checkpoint/config.yaml)"
        },
    )

    # Output directory for evaluation results
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results (defaults to checkpoint_path/eval_output)"},
    )

    # Custom evaluation configuration
    custom_eval: CustomEvaluationConfig = field(
        default_factory=CustomEvaluationConfig,
        metadata={"help": "Custom evaluation configuration (reused from experiment_configs)"},
    )
