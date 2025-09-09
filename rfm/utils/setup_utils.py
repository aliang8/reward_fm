#!/usr/bin/env python3
"""
Shared setup utilities for RFM training.
This file contains setup functions that can be reused across different training scripts.
"""

from dataclasses import asdict
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLModel,
    TrainingArguments,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoImageProcessor,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig
from typing import Tuple, Optional, Union

from rfm.models.rfm import RFMModel
from rfm.models.rfm_vqa import RFMModelVQA
from rfm.models.rfm_transformer import RFMConfig, RFMTransformer
from rfm.data.generators.generator import DataGenerator
from rfm.data.generators.vqa_generator import VQADataGenerator
from rfm.data.batch_collator import BatchCollator
from rfm.data.dataset import InfiniteDataGeneratorDataset
from rfm.data.generators.success_failure import PairedSuccessFailureGenerator
from rfm.data.generators.reward_alignment import RewardAlignmentGenerator
from rfm.data.generators.confusion_matrix import ConfusionMatrixGenerator
from rfm.data.generators.wrong_task import WrongTaskGenerator
from rfm.data.generators.progress import ProgressGenerator
from rfm.utils.logging import rank_0_print
from rfm.configs.experiment_configs import ExperimentConfig, ModelConfig
from rfm.data.vqa_batch_collator import VQABatchCollator
from rfm.data.rewind_batch_collator import ReWiNDBatchCollator


def setup_model_and_processor(cfg: ModelConfig, hf_model_id: str = "") -> Tuple[AutoProcessor, RFMModel]:
    """Shared function to set up model, processor, and tokenizer for both training and evaluation"""

    # Get current rank for logging
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        rank_0_print(f"Setting up model and processor on rank {rank}...")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=cfg.trust_remote_code,
        # temporal_patch_size=1,
        # fps=1,
        # num_frames=cfg.data.max_frames,
        do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
        # max_frames=cfg.data.max_frames,
    )

    rank_0_print(f"Processor: {processor}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Create a fresh model instance
    base_model = Qwen2_5_VLModel.from_pretrained(cfg.base_model_id)

    # Add RFM special tokens if they don't exist
    special_tokens = ["<|split_token|>", "<|reward_token|>", "<|pref_token|>"]
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            if rank == 0:
                rank_0_print(f"Added special token: {token}")

    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        if rank == 0:
            rank_0_print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(processor.tokenizer)}")
        base_model.resize_token_embeddings(len(processor.tokenizer))
        if rank == 0:
            rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")

    # Initialize RFM model wrapper with the pre-loaded base model
    if rank == 0:
        rank_0_print(f"Initializing RFM model on rank {rank}...")
    rfm_model = RFMModel(config=base_model.config, processor=processor, base_model=base_model)

    if hf_model_id:
        rank_0_print(f"Loading model from {hf_model_id} on rank {rank}")

        # before = rfm_model.model.visual.blocks[0].mlp.down_proj.weight
        # before = rfm_model.preference_head.weight
        # load the model from the evaluation path
        rfm_model = RFMModel.from_pretrained(hf_model_id, processor=processor, base_model=base_model)

    # Only print model architecture on rank 0
    if rank == 0:
        rank_0_print(f"Model architecture initialized on rank {rank}")

    return processor, rfm_model


def setup_peft_model(rfm_model: RFMModel, cfg: ExperimentConfig) -> RFMModel:
    """Shared function to apply PEFT configuration to the model"""

    if cfg.peft.use_peft:
        rank_0_print("Using PEFT/LoRA training...")
        lora_config = LoraConfig(
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=cfg.peft.target_modules,
            lora_dropout=cfg.peft.lora_dropout,
            bias=cfg.peft.bias,
        )
        if cfg.peft.peft_vision_encoder:
            # vision backbone is frozen, but we can still train the LoRA parameters
            rank_0_print("Attaching LoRA to only the vision encoder...")
            rfm_model.base_model.model.visual = get_peft_model(rfm_model.base_model.model.visual, lora_config)
    else:
        rank_0_print("Using full model training (no PEFT)...")

    # Configure which parts of the model to train based on config
    for name, param in rfm_model.named_parameters():
        # Train prediction heads based on individual settings
        if "progress_head" in name:
            param.requires_grad = cfg.model.train_progress_head
        elif "preference_head" in name:
            param.requires_grad = cfg.model.train_preference_head
        elif "similarity_head" in name:
            param.requires_grad = cfg.model.train_similarity_head
        # Train vision encoder if specified
        elif "visual" in name or "vision" in name:
            # if PEFT enabled, we don't need to do anything
            if cfg.peft.use_peft and cfg.peft.peft_vision_encoder:
                pass 
            else:
                param.requires_grad = cfg.model.train_vision_encoder
        elif "language_model" in name:
            param.requires_grad = cfg.model.train_language_model
        else:
            param.requires_grad = False

    if cfg.logging.print_trainable_parameters:
        # Count trainable parameters manually - defer printing until after FSDP setup
        trainable_params = sum(p.numel() for p in rfm_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in rfm_model.parameters())
        rank_0_print(
            f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
        )
        rank_0_print(f"Training configuration:")
        rank_0_print(f"  - Vision encoder: {cfg.model.train_vision_encoder}")
        rank_0_print(f"  - Language model: {cfg.model.train_language_model}")
        rank_0_print(f"  - Progress head: {cfg.model.train_progress_head}")
        rank_0_print(f"  - Preference head: {cfg.model.train_preference_head}")
        rank_0_print(f"  - Similarity head: {cfg.model.train_similarity_head}")

    return rfm_model


def create_training_arguments(cfg: ExperimentConfig, output_dir: str, is_eval: bool = False) -> TrainingArguments:
    """Shared function to create TrainingArguments for both training and evaluation"""

    # Base arguments that are the same for both training and evaluation
    base_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "save_strategy": cfg.training.save_strategy,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "bf16": cfg.training.bf16,
        "fp16": cfg.training.fp16,
        "remove_unused_columns": cfg.training.remove_unused_columns,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "dataloader_pin_memory": cfg.data.dataloader_pin_memory,
        "dataloader_num_workers": cfg.data.dataloader_num_workers,
        "save_safetensors": True,
        "save_total_limit": 2,
        # Evaluation settings
        "eval_strategy": cfg.training.evaluation_strategy,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "do_eval": cfg.training.do_eval,
        "prediction_loss_only": cfg.training.prediction_loss_only,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "warmup_steps": cfg.training.warmup_steps,
        "warmup_ratio": cfg.training.warmup_ratio,
        "max_grad_norm": cfg.training.max_grad_norm,
    }

    # Add eval_steps if evaluation_strategy is "steps"
    if cfg.training.evaluation_strategy == "steps" and cfg.training.eval_steps is not None:
        base_args["eval_steps"] = cfg.training.eval_steps

    if is_eval:
        # Evaluation-specific arguments
        base_args.update(
            {
                "per_device_eval_batch_size": 2,
                "num_train_epochs": -1,
                "max_steps": 1,
                "report_to": "none",
            }
        )
    else:
        # Training-specific arguments
        base_args.update(
            {
                "num_train_epochs": cfg.training.num_train_epochs if cfg.training.num_train_epochs is not None else 1,
                "max_steps": cfg.training.max_steps if cfg.training.max_steps is not None else -1,
                "report_to": ["wandb"] if cfg.logging.use_wandb else [],
            }
        )

    return TrainingArguments(**base_args)


def setup_data_generator(cfg: ExperimentConfig, is_eval: bool = False) -> DataGenerator:
    """Shared function to create DataGenerator for training or evaluation"""

    # Get current rank for logging
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        rank_0_print(f"Setting up data generator on rank {rank} for {'evaluation' if is_eval else 'training'}...")

    if is_eval:
        datasets = cfg.data.eval_datasets
        subsets = cfg.data.eval_subsets
    else:
        datasets = cfg.data.train_datasets
        subsets = cfg.data.train_subsets

    # Validate that train_datasets and train_subsets have the same length
    if len(datasets) != len(subsets):
        raise ValueError(
            f"datasets and subsets must have the same length. Got {len(datasets)} datasets and {len(subsets)} subsets"
        )

    if rank == 0:
        rank_0_print(f"Loading {len(datasets)} datasets with corresponding subsets")
        for i, (dataset, subset) in enumerate(zip(datasets, subsets)):
            rank_0_print(f"  Dataset {i + 1}: {dataset} -> {subset}")

    if cfg.data.model_type == "vqa":
        if cfg.data.dataset_type == "reward_alignment":
            data_generator = RewardAlignmentGenerator(config=cfg.data, is_evaluation=is_eval)
        elif cfg.data.dataset_type == "success_failure":
            data_generator = PairedSuccessFailureGenerator(config=cfg.data, is_evaluation=is_eval)
        elif cfg.data.dataset_type == "policy_ranking":
            data_generator = ProgressGenerator(config=cfg.data, is_evaluation=is_eval)
        else:
            data_generator = VQADataGenerator(config=cfg.data, is_evaluation=is_eval)
    else:
        if cfg.data.dataset_type == "reward_alignment":
            data_generator = RewardAlignmentGenerator(config=cfg.data, is_evaluation=is_eval)
        elif cfg.data.dataset_type == "success_failure":
            data_generator = PairedSuccessFailureGenerator(config=cfg.data, is_evaluation=is_eval)
        elif cfg.data.dataset_type == "policy_ranking":
            data_generator = ProgressGenerator(config=cfg.data, is_evaluation=is_eval)
        elif cfg.data.dataset_type == "confusion_matrix":
            data_generator = ConfusionMatrixGenerator(
                config=cfg.data, is_evaluation=is_eval, max_trajectories=cfg.data.max_trajectories
            )
        elif cfg.data.dataset_type == "wrong_task":
            data_generator = WrongTaskGenerator(
                config=cfg.data, is_evaluation=is_eval, max_trajectories=cfg.data.max_trajectories
            )
        else:
            data_generator = DataGenerator(config=cfg.data, is_evaluation=is_eval)

    if rank == 0:
        rank_0_print(f"Data generator initialized on rank {rank}")

    return data_generator


def setup_dataset(
    data_generator: DataGenerator, dataset_type: str = "default", dataset_kwargs: dict = {}
) -> InfiniteDataGeneratorDataset:
    """Shared function to create training or evaluation dataset based on config"""

    # Get the dataset type from the data generator config
    config_dataset_type = data_generator.config.dataset_type

    rank_0_print(f"Setting up {dataset_type} dataset with type: {config_dataset_type}")
    dataset = InfiniteDataGeneratorDataset(data_generator, **dataset_kwargs)

    rank_0_print(f"{dataset_type.capitalize()} dataset created successfully with {len(dataset)} samples")
    return dataset


def setup_eval_dataset(cfg: ExperimentConfig) -> InfiniteDataGeneratorDataset:
    """Create evaluation dataset using eval-specific configuration"""

    # Create evaluation data generator
    eval_data_generator = setup_data_generator(cfg, is_eval=True)

    # Create evaluation dataset
    eval_dataset = setup_dataset(
        eval_data_generator,
        dataset_type=cfg.data.dataset_type,
        dataset_kwargs={"max_samples": cfg.data.eval_subset_size, "num_bins": cfg.data.num_bins, "fps": cfg.data.fps},
    )

    return eval_dataset


def setup_batch_collator(processor: AutoProcessor, cfg: ExperimentConfig) -> BatchCollator:
    """Shared function to create BatchCollator"""

    rank_0_print("Setting up batch collator...")

    batch_collator = BatchCollator(
        processor=processor,
        max_length=cfg.training.max_seq_length,
        resized_height=cfg.data.resized_height,
        resized_width=cfg.data.resized_width,
    )

    rank_0_print("Batch collator created successfully")
    return batch_collator


def setup_vqa_model_and_processor(cfg: ModelConfig, hf_model_id: str = ""):
    """Setup VQA baseline model and processor from a VQA-specific config."""
    # Get current rank for logging
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        rank_0_print(f"Setting up model and processor on rank {rank}...")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=cfg.trust_remote_code,
        # temporal_patch_size=1,
        # fps=1,
        # num_frames=cfg.data.max_frames,
        do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
        # max_frames=cfg.data.max_frames,
        padding_side="left",
        cache_dir="/scr/ykorkmaz/rfm/model/base_qwen",
    )

    rank_0_print(f"Processor: {processor}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load base conditional generation model for VQA
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_id, cache_dir="/scr/ykorkmaz/rfm/model/base_qwen"
    )

    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        if rank == 0:
            rank_0_print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(processor.tokenizer)}")
        base_model.resize_token_embeddings(len(processor.tokenizer))
        base_model.config.vocab_size = len(processor.tokenizer)
        if rank == 0:
            rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")

    # Initialize RFM model wrapper with the pre-loaded base model
    if rank == 0:
        rank_0_print(f"Initializing RFM-VQA model on rank {rank}...")
    rfm_model = RFMModelVQA(config=base_model.config, processor=processor, base_model=base_model)

    if hf_model_id:
        rank_0_print(f"Loading model from {hf_model_id} on rank {rank}")

        # before = rfm_model.model.visual.blocks[0].mlp.down_proj.weight
        # before = rfm_model.preference_head.weight
        # load the model from the evaluation path
        rfm_model = RFMModelVQA.from_pretrained(hf_model_id, processor=processor, base_model=base_model)

    # Only print model architecture on rank 0
    if rank == 0:
        rank_0_print(f"Model architecture initialized on rank {rank}")

    return processor, rfm_model


def setup_transformer_model_and_processor(cfg: ModelConfig, hf_model_id: str = ""):
    """Shared function to set up model, processor, and tokenizer for both training and evaluation"""
    'The data should be same with RFM model'
    'The model should be different, should be the transformer model'
    # Get current rank for logging
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        rank_0_print(f"Setting up model and processor on rank {rank}...")

    # Load processor and tokenizer 
    # here should change to DInov2 processor maybe correct here?
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)

    rank_0_print(f"Processor: {processor}")

    # then here we should load minilm tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    ) # here we finish init the tokenizer and processor
    
    # Initialize RFM model wrapper with the pre-loaded base model
    if rank == 0:
        rank_0_print(f"Initializing ReWiND model on rank {rank}...")
    rfm_config = RFMConfig(**asdict(cfg.rewind))
    rfm_model = RFMTransformer(config=rfm_config)
    # Not support loading from hf for now cuz we are training from scratch
    '''
    Maybe add pre-trained weight here in the future
    '''
    # load the model from the evaluation path
    if hf_model_id:
        rank_0_print(f"Loading model from {hf_model_id} on rank {rank}")

        # before = rfm_model.model.visual.blocks[0].mlp.down_proj.weight
        # before = rfm_model.preference_head.weight
        # load the model from the evaluation path
        state_dict = torch.load(hf_model_id, map_location='cpu')
        rfm_model.load_state_dict(state_dict)

    # Only print model architecture on rank 0
    if rank == 0:
        rank_0_print(f"Model architecture initialized on rank {rank}")

    return tokenizer, processor, rfm_model






def setup_vqa_batch_collator(processor: AutoProcessor, cfg: ExperimentConfig) -> VQABatchCollator:
    """Create VQA batch collator using VQA config."""
    rank_0_print("Setting up VQA batch collator...")
    collator = VQABatchCollator(
        processor=processor,
        max_length=cfg.training.max_seq_length,
        resized_height=cfg.data.resized_height,
        resized_width=cfg.data.resized_width,
    )
    rank_0_print("VQA batch collator created successfully")
    return collator


def setup_rewind_batch_collator(processor: AutoProcessor, tokenizer: AutoTokenizer, cfg: ExperimentConfig) -> BatchCollator:
    """Shared function to create BatchCollator"""

    rank_0_print("Setting up batch collator...")
    batch_collator = ReWiNDBatchCollator(
        processor=processor,
        tokenizer = tokenizer,
        max_length=cfg.training.max_seq_length,
        resized_height=cfg.data.resized_height,
        resized_width=cfg.data.resized_width,
    )

    rank_0_print("Batch collator created successfully")
    return batch_collator