#!/usr/bin/env python3
"""
Experiment configurations for RFM training.
This file contains all the dataclass configurations that can be reused
across different training scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Config for model settings"""
    base_model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)


@dataclass
class PEFTConfig:
    """Config for PEFT/LoRA settings"""
    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT/LoRA or train full model"})
    r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default="none")
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Additional options for more comprehensive training
    train_vision_encoder: bool = field(default=False, metadata={"help": "Whether to train the vision encoder"})
    train_language_model: bool = field(default=True, metadata={"help": "Whether to train the language model"})
    train_value_head: bool = field(default=True, metadata={"help": "Whether to train the value head"})
    # RFM-specific head training options
    train_progress_head: bool = field(default=True, metadata={"help": "Whether to train the progress prediction head"})
    train_preference_head: bool = field(default=True, metadata={"help": "Whether to train the preference prediction head"})
    train_similarity_head: bool = field(default=True, metadata={"help": "Whether to train the similarity scoring head"})


@dataclass
class DataConfig:
    """Config for data settings"""
    # Dataset paths and sources
    dataset_path: str = field(default="aliangdw/rfm")
    dataset_subsets: List[str] = field(default_factory=lambda: ["libero_90"])
    base_dir: str = field(default="libero_dpo_dataset")
    
    # Video processing settings
    max_frames: int = field(default=32)  # Maximum frames per trajectory
    video_frame_sampling: str = field(default="uniform")  # "uniform", "random", "first", "middle"
    resized_height: int = field(default=128, metadata={"help": "Height to resize images/videos to"})
    resized_width: int = field(default=128, metadata={"help": "Width to resize images/videos to"})
    
    # Data generation settings
    preference_ratio: float = field(default=0.5)
    similarity_ratio: float = field(default=0.5)
    dataset_preference_ratio: float = field(default=0.7)
    shuffle: bool = field(default=True)
    seed: int = field(default=42)
    num_proc: int = field(default=1)
    force_reprocess: bool = field(default=False, metadata={"help": "Force reprocessing of dataset even if cached version exists"})
    
    # Data loading settings
    dataloader_pin_memory: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)


@dataclass
class TrainingConfig:
    """Config for training settings"""
    # Hardware settings
    num_gpus: int = field(default=2, metadata={"help": "Number of GPUs to use for training"})
    
    # Output and logging
    output_dir: str = field(default="./rfm_model_output")
    max_seq_length: int = field(default=1024)
    beta: float = field(default=0.1)
    resume_from_checkpoint: Optional[str] = field(default=None)
    
    # Training arguments
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=5e-7)
    num_train_epochs: Optional[int] = field(default=1)  # Default to 1 epoch if not specified
    save_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    bf16: bool = field(default=False)
    fp16: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_bucket_cap_mb: int = field(default=25)
    max_steps: Optional[int] = field(default=-1)  # -1 means no limit, use num_train_epochs instead
    save_steps: int = field(default=100)
    
    # FSDP configuration
    fsdp_strategy: str = field(default="fsdp2", metadata={"help": "FSDP strategy: 'fsdp' or 'fsdp2'"})


@dataclass
class LoggingConfig:
    """Config for logging settings"""
    print_trainable_parameters: bool = field(default=True)
    save_model: bool = field(default=True)
    save_processor: bool = field(default=True)
    # Wandb configuration
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Weights & Biases logging"})
    wandb_project: str = field(default="rfm-model", metadata={"help": "Wandb project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity/username"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Wandb run name"})


@dataclass
class EvaluationConfig:
    """Config for evaluation settings"""
    model_path: str = field(default="./rfm_model_output")
    eval_subset_size: int = field(default=10, metadata={"help": "Number of examples to use for evaluation"})
    eval_dataset_path: str = field(default="aliangdw/rfm")
    eval_base_dir: str = field(default="libero_dpo_dataset")
    eval_dataset_subsets: List[str] = field(default_factory=lambda: ["libero_90"])


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'evaluate'"})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode"})
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig) 