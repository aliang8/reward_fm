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

from rfm.configs.experiment_configs import CustomEvaluationConfig, DataConfig


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

    # Reward model: "gvl", "vlac", "rlvlmf", "rfm", "rewind", or "roboreward"
    reward_model: str = field(
        default="rlvlmf",
        metadata={
            "help": "Reward model: 'gvl' or 'vlac' for progress, 'rlvlmf' for preference, 'rfm' or 'rewind' for trained models, 'roboreward' for RoboReward baseline"
        },
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
    vlac_use_images: bool = field(
        default=False,
        metadata={
            "help": "If True, use image mode (get_trajectory_critic). If False, use video mode (web_trajectory_critic)"
        },
    )

    # RFM/ReWiND settings (only used if reward_model is "rfm" or "rewind")
    rfm_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to RFM/ReWiND model checkpoint (HuggingFace repo ID or local path, required for rfm/rewind)"
        },
    )
    rfm_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for RFM/ReWiND model inference"},
    )

    # RoboReward settings (only used if reward_model is "roboreward")
    roboreward_model_path: Optional[str] = field(
        default="teetone/RoboReward-4B",
        metadata={
            "help": "Path to RoboReward model (HuggingFace repo ID, e.g., 'teetone/RoboReward-8B' or 'teetone/RoboReward-4B')"
        },
    )
    roboreward_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate for RoboReward"},
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
