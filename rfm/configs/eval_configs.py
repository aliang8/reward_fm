#!/usr/bin/env python3
"""
Evaluation configuration for RFM.
This file contains the EvaluationConfig dataclass for evaluation-specific parameters.
"""

from dataclasses import dataclass, field

from rfm.configs.experiment_configs import DataConfig


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

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

    # Reuse DataConfig for data settings
    data: DataConfig = field(
        default_factory=DataConfig, metadata={"help": "Data configuration (reused from experiment_configs)"}
    )
