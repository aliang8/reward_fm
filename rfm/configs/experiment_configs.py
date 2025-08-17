#!/usr/bin/env python3
"""
Experiment configurations for RFM training.
This file contains all the dataclass configurations that can be reused
across different training scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    """Config for model settings"""

    base_model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)
    train_vision_encoder: bool = field(default=False, metadata={"help": "Whether to train the vision encoder"})
    train_language_model: bool = field(default=True, metadata={"help": "Whether to train the language model"})
    train_value_head: bool = field(default=True, metadata={"help": "Whether to train the value head"})
    # RFM-specific head training options
    train_progress_head: bool = field(default=True, metadata={"help": "Whether to train the progress prediction head"})
    train_preference_head: bool = field(
        default=True, metadata={"help": "Whether to train the preference prediction head"}
    )
    train_similarity_head: bool = field(default=True, metadata={"help": "Whether to train the similarity scoring head"})


@dataclass
class PEFTConfig:
    """Config for PEFT/LoRA settings"""

    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT/LoRA or train full model"})
    r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default="none")
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    # Dataset paths and subsets
    train_datasets: List[str] = field(
        default_factory=lambda: ["abraranwar/libero_rfm"], metadata={"help": "List of training dataset names"}
    )
    train_subsets: List[List[str]] = field(
        default_factory=lambda: [["libero_90"]], metadata={"help": "List of training dataset subsets"}
    )
    eval_datasets: List[str] = field(
        default_factory=lambda: ["abraranwar/libero_rfm"], metadata={"help": "List of evaluation dataset names"}
    )
    eval_subsets: List[List[str]] = field(
        default_factory=lambda: [["libero_10"]], metadata={"help": "List of evaluation dataset subsets"}
    )

    # Dataset type and configuration
    dataset_type: str = field(
        default="preference",
        metadata={"help": "Dataset type: 'preference', 'similarity', 'paired_video', 'rewound', 'success_failure'"},
    )

    # Rewound dataset specific parameters
    # Example rewound config:
    # dataset_type: "rewound"
    # rewind_lengths: [1, 2, 4, 8]  # Generate rewinds of 1, 2, 4, and 8 frames
    # samples_per_trajectory: 2  # Generate 2 preference samples per trajectory
    # Note: Original trajectories are preferred over rewound versions
    rewind_lengths: Optional[List[int]] = field(
        default=None, metadata={"help": "List of rewind lengths for rewound dataset (default: 1 to max_frames)"}
    )
    samples_per_trajectory: int = field(
        default=1, metadata={"help": "Number of preference samples to generate per trajectory for rewound dataset"}
    )

    # Success-failure dataset specific parameters
    # Example success_failure config:
    # dataset_type: "success_failure"
    # Note: Generates ALL possible pairs between successful and failed trajectories for each task
    # Note: Successful trajectories are preferred over failed versions of the same task

    # Video processing parameters
    max_frames: int = field(default=8, metadata={"help": "Maximum number of frames to extract from videos"})
    video_frame_sampling: str = field(
        default="uniform", metadata={"help": "Frame sampling strategy: 'uniform', 'random', 'start', 'end'"}
    )
    resized_height: int = field(default=224, metadata={"help": "Height to resize video frames to"})
    resized_width: int = field(default=224, metadata={"help": "Width to resize video frames to"})

    # Data generation parameters
    preference_ratio: float = field(default=0.7, metadata={"help": "Ratio of preference samples to similarity samples"})
    dataset_preference_ratio: float = field(
        default=0.8, metadata={"help": "Ratio of dataset preference samples to generated preference samples"}
    )

    # Processing parameters
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the dataset"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    num_proc: int = field(default=1, metadata={"help": "Number of processes for dataset processing"})
    force_reprocess: bool = field(
        default=False, metadata={"help": "Force reprocessing of datasets even if cache exists"}
    )

    # Evaluation parameters
    eval_subset_size: Optional[int] = field(default=100, metadata={"help": "Number of samples to use for evaluation"})

    # Dataloader parameters
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin memory in dataloader"})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of worker processes for dataloader"})


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

    # Evaluation settings
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'"})
    eval_steps: Optional[int] = field(
        default=None, metadata={"help": "Number of steps between evaluations (required if evaluation_strategy='steps')"}
    )
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation during training"})
    prediction_loss_only: bool = field(default=True, metadata={"help": "Only compute loss for the prediction head"})

    # Optimizer settings
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=0)
    warmup_ratio: float = field(default=0.1)


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

    model_path: Optional[str] = field(default=None)


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
