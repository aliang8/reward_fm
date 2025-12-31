#!/usr/bin/env python3
"""
Evaluation configuration for RFM.
This file contains evaluation configuration classes:
- EvalServerConfig: For evaluation server runs (eval_server.py)
- OfflineEvalConfig: For standalone evaluation runs (run_eval_only.py)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from hydra.core.config_store import ConfigStore

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
        default=1, metadata={"help": "Max worker threads (None for auto, typically same as num_gpus)"}
    )

    # Evaluation parameters
    batch_size: int = field(default=4, metadata={"help": "Batch size for evaluation"})
    server_url: str = field(default="0.0.0.0", metadata={"help": "Evaluation server URL"})
    server_port: int = field(default=8000, metadata={"help": "Evaluation server port"})


@dataclass
class OfflineEvalConfig:
    """Configuration for standalone evaluation runs (run_eval_only.py)."""

    # Model path (HuggingFace model ID or local checkpoint path)
    model_path: str = field(
        default="",
        metadata={"help": "HuggingFace model ID (e.g., 'aliangdw/rfm_model') or local checkpoint path"},
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

    def __post_init__(self):
        """Convert nested dict configs to dataclass instances."""
        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)


@dataclass
class BaselineEvalConfig:
    """Configuration for baseline evaluation runs (run_baseline_eval.py)."""

    # Baseline type: "gvl" or "vlac" for progress evaluation or "rlvlmf" for preference evaluation
    baseline_type: str = field(
        default="rlvlmf",
        metadata={"help": "Baseline type: 'gvl' or 'vlac' for progress, 'rlvlmf' for preference"},
    )

    # VLM provider for RL-VLM-F
    vlm_provider: str = field(
        default="gemini",
        metadata={"help": "VLM provider for RL-VLM-F: 'gemini' or 'openai'"},
    )

    # RL-VLM-F settings
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for RL-VLM-F"},
    )

    # GVL settings
    gvl_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "GVL API key (defaults to GEMINI_API_KEY env var)"},
    )
    gvl_max_frames: int = field(
        default=15,
        metadata={"help": "Maximum frames for GVL"},
    )
    gvl_offset: float = field(
        default=0.5,
        metadata={"help": "Frame offset for GVL"},
    )

    # VLAC settings
    vlac_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to VLAC model checkpoint (required for vlac baseline)"},
    )
    vlac_device: str = field(
        default="cuda:0",
        metadata={"help": "Device for VLAC model"},
    )
    vlac_model_type: str = field(
        default="internvl2",
        metadata={"help": "VLAC model type"},
    )
    vlac_temperature: float = field(
        default=0.5,
        metadata={"help": "Temperature for VLAC"},
    )
    vlac_batch_num: int = field(
        default=5,
        metadata={"help": "Batch number for VLAC processing"},
    )
    vlac_skip: int = field(
        default=5,
        metadata={"help": "Pair-wise step size for VLAC"},
    )
    vlac_frame_skip: bool = field(
        default=True,
        metadata={"help": "Whether to skip frames for VLAC efficiency"},
    )

    # Custom evaluation configuration
    custom_eval: CustomEvaluationConfig = field(
        default_factory=CustomEvaluationConfig,
        metadata={"help": "Custom evaluation configuration"},
    )

    # Output directory for evaluation results
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results"},
    )

    def __post_init__(self):
        """Convert nested dict configs to dataclass instances."""
        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)


# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="eval_server_config", node=EvalServerConfig)
cs.store(name="eval_only_config", node=OfflineEvalConfig)
cs.store(name="baseline_eval_config", node=BaselineEvalConfig)
