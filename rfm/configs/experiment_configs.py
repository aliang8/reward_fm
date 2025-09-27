#!/usr/bin/env python3
"""
Experiment configurations for RFM training.
This file contains all the dataclass configurations that can be reused
across different training scripts.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import PretrainedConfig


@dataclass
class ModelConfig(PretrainedConfig):
    """Config for model settings"""

    base_model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_type: str = field(default="default")
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

    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT/LoRA or train full model"})
    peft_vision_encoder: bool = field(default=False, metadata={"help": "Whether to attach LoRA to the vision encoder"})

    # rewind sub-config
    rewind: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        from rfm.models.rewind_transformer import ReWINDTransformerConfig

        if self.rewind is not None and isinstance(self.rewind, dict):
            self.rewind = ReWINDTransformerConfig(**self.rewind)


@dataclass
class PEFTConfig:
    """Config for PEFT/LoRA settings"""

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
        default="default",
        metadata={"help": "Dataset type: 'default', 'rewound', 'success_failure'"},
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

    max_frames: int = field(default=8, metadata={"help": "Maximum number of frames to extract from videos"})
    resized_height: int = field(default=224, metadata={"help": "Height to resize video frames to"})
    resized_width: int = field(default=224, metadata={"help": "Width to resize video frames to"})

    # Data generation parameters
    sample_type_ratio: List[float] = field(
        default_factory=lambda: [1, 1, 1], metadata={"help": "Ratio of pref, progress and similarity samples"}
    )
    dataset_preference_ratio: float = field(
        default=0.8, metadata={"help": "Ratio of dataset preference samples to generated preference samples"}
    )
    # Tunable strategy ratios for preference negative generation: [rewind, suboptimal_same_task, different_task, video_binned]
    preference_strategy_ratio: List[float] = field(default_factory=lambda: [1, 1, 1, 1])
    # Tunable strategy ratios for progress generation: [default, rewind_same_task, different_task]
    progress_strategy_ratio: List[float] = field(default_factory=lambda: [1, 1, 1])
    data_source_weights: Optional[Dict[str, float]] = field(
        default=None, 
        metadata={"help": "Dictionary mapping data source names to sampling weights (e.g., {'metaworld': 0.2, 'libero': 0.8})"}
    )

    # Processing parameters
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the dataset"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    # Evaluation parameters
    eval_subset_size: Optional[int] = field(default=100, metadata={"help": "Number of samples to use for evaluation"})

    # Dataloader parameters
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin memory in dataloader"})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of worker processes for dataloader"})

    # Video binned dataset specific parameters
    num_bins: int = field(default=10, metadata={"help": "Number of bins to use for video binned dataset"})
    fps: int = field(default=10, metadata={"help": "Frames per second to extract from videos"})

    max_trajectories: int = field(default=-1, metadata={"help": "Maximum number of trajectories to use for dataset"})
    n_wrong_tasks: int = field(
        default=5, metadata={"help": "Number of wrong tasks to use for wrong task preference dataset"}
    )

    # Embedding loading parameters
    load_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to load precomputed embeddings instead of processing frames (ReWiND only)"},
    )
    
    # Data source weighting parameters
    data_source_weights: Optional[Dict[str, float]] = field(
        default=None, 
        metadata={"help": "Dictionary mapping data source names to sampling weights (e.g., {'metaworld': 0.2, 'libero': 0.8})"}
    )


@dataclass
class CustomEvaluationConfig:
    """Config for custom evaluation settings"""

    eval_types: List[str] = field(default_factory=lambda: ["policy_ranking", "confusion_matrix", "reward_alignment"])
    policy_ranking: List[List[str]] = field(default_factory=lambda: [["aliangdw/metaworld_rfm", "metaworld"]])
    confusion_matrix: List[List[str]] = field(default_factory=lambda: [["aliangdw/metaworld_rfm", "metaworld"]])
    reward_alignment: List[List[str]] = field(
        default_factory=lambda: [["HenryZhang/metaworld_rewind_rfm_eval", "metaworld_rewind_eval"]]
    )


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
    custom_eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps between custom evaluations (required if evaluation_strategy='steps')"},
    )
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation during training"})
    prediction_loss_only: bool = field(default=True, metadata={"help": "Only compute loss for the prediction head"})

    # Optimizer settings
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=0)
    warmup_ratio: float = field(default=0.1)
    max_grad_norm: float = field(default=1.0)

    # RFM specific settings
    predict_pref_progress: bool = field(
        default=False, metadata={"help": "Whether to predict progress for preference samples"}
    )


@dataclass
class SaveBestConfig:
    """Configuration for SaveBestCallback"""
    
    # Metric monitoring
    metric_name: str = field(default="custom_eval/p_rank_spearman_mw", metadata={"help": "Metric name to monitor for saving best models"})
    greater_is_better: bool = field(default=True, metadata={"help": "Whether higher values are better for the metric"})
    keep_top_k: int = field(default=5, metadata={"help": "Number of best checkpoints/uploads to keep"})
    
    # Hub upload configuration
    upload_to_hub: bool = field(default=False, metadata={"help": "Whether to upload best models to HuggingFace Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "HuggingFace model ID (username/model-name)"})
    hub_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace token (or set HF_TOKEN env var)"})
    hub_private: bool = field(default=False, metadata={"help": "Whether to make the Hub model private"})


@dataclass
class LoggingConfig:
    """Config for logging settings"""

    save_model: bool = field(default=True)
    save_processor: bool = field(default=True)
    # Wandb configuration
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Weights & Biases logging"})
    wandb_project: str = field(default="rfm-model", metadata={"help": "Wandb project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity/username"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Wandb run name"})
    
    # SaveBest configuration
    save_best: Optional[SaveBestConfig] = field(default=None, metadata={"help": "SaveBestCallback configuration"})


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""

    mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'evaluate'"})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode"})
    trainer_cls: str = field(
        default="rfm_heads", metadata={"help": "Trainer class: 'rfm_heads', 'rewind_transformer', 'rfm_vqa'"}
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    custom_eval: CustomEvaluationConfig = field(default_factory=CustomEvaluationConfig)

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)

        if isinstance(self.peft, dict):
            self.peft = PEFTConfig(**self.peft)

        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)

        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
            
        # Handle nested SaveBestConfig in LoggingConfig
        if hasattr(self.logging, 'save_best') and isinstance(self.logging.save_best, dict):
            self.logging.save_best = SaveBestConfig(**self.logging.save_best)
        elif self.logging.save_best is None:
            # Set default SaveBestConfig if not provided
            self.logging.save_best = SaveBestConfig()

        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)
