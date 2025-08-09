#!/usr/bin/env python3
"""
Shared setup utilities for RFM training.
This file contains setup functions that can be reused across different training scripts.
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLModel, TrainingArguments
from peft import get_peft_model, LoraConfig
from typing import Tuple, Optional

from rfm.models.rfm import RFMModel
from rfm.data.data_generator import DataGenerator, InfiniteDataGeneratorDataset, BatchCollator
from rfm.utils.logging import rank_0_print
from rfm.configs.experiment_configs import ExperimentConfig


def setup_model_and_processor(cfg: ExperimentConfig) -> Tuple[AutoProcessor, RFMModel]:
    """Shared function to set up model, processor, and tokenizer for both training and evaluation"""
    
    # Get current rank for logging
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        rank_0_print(f"Setting up model and processor on rank {rank}...")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        cfg.model.base_model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Create a fresh model instance
    base_model = Qwen2_5_VLModel.from_pretrained(cfg.model.base_model_id)

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
    rfm_model = RFMModel(
        config=base_model.config, tokenizer=processor.tokenizer, base_model=base_model
    )

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
        peft_rfm_model = get_peft_model(rfm_model, lora_config)
        for name, param in peft_rfm_model.named_parameters():
            if any(head in name for head in ["progress_head", "preference_head", "similarity_head"]):
                param.requires_grad = True
        return peft_rfm_model
    else:
        rank_0_print("Using full model training (no PEFT)...")
        peft_rfm_model = rfm_model
        # Configure which parts of the model to train based on config
        for name, param in peft_rfm_model.named_parameters():
            # Train prediction heads based on individual settings
            if "progress_head" in name:
                param.requires_grad = cfg.peft.train_progress_head
            elif "preference_head" in name:
                param.requires_grad = cfg.peft.train_preference_head
            elif "similarity_head" in name:
                param.requires_grad = cfg.peft.train_similarity_head
            # Train vision encoder if specified
            elif "visual" in name or "vision" in name:
                param.requires_grad = cfg.peft.train_vision_encoder
            # Train language model if specified
            elif "model" in name and not ("visual" in name or "vision" in name):
                param.requires_grad = cfg.peft.train_language_model
            # Default: train if language model training is enabled
            else:
                param.requires_grad = cfg.peft.train_language_model
        
        if cfg.logging.print_trainable_parameters:
            # Count trainable parameters manually - defer printing until after FSDP setup
            trainable_params = sum(p.numel() for p in peft_rfm_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in peft_rfm_model.parameters())
            rank_0_print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
            rank_0_print(f"Training configuration:")
            rank_0_print(f"  - Vision encoder: {cfg.peft.train_vision_encoder}")
            rank_0_print(f"  - Language model: {cfg.peft.train_language_model}")
            rank_0_print(f"  - Progress head: {cfg.peft.train_progress_head}")
            rank_0_print(f"  - Preference head: {cfg.peft.train_preference_head}")
            rank_0_print(f"  - Similarity head: {cfg.peft.train_similarity_head}")

        return peft_rfm_model


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
    }
    
    # Add eval_steps if evaluation_strategy is "steps"
    if cfg.training.evaluation_strategy == "steps" and cfg.training.eval_steps is not None:
        base_args["eval_steps"] = cfg.training.eval_steps
    
    if is_eval:
        # Evaluation-specific arguments
        base_args.update({
            "per_device_eval_batch_size": 2,
            "num_train_epochs": -1,
            "max_steps": 1,
            "report_to": "none",
        })
    else:
        # Training-specific arguments
        base_args.update({
            "num_train_epochs": cfg.training.num_train_epochs if cfg.training.num_train_epochs is not None else 1,
            "max_steps": cfg.training.max_steps if cfg.training.max_steps is not None else -1,
            "report_to": ["wandb"] if cfg.logging.use_wandb else [],
        })
    
    return TrainingArguments(**base_args)


def setup_data_generator(cfg: ExperimentConfig) -> DataGenerator:
    """Shared function to create DataGenerator for training or evaluation"""
    
    # Get current rank for logging
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        rank_0_print(f"Setting up data generator on rank {rank}...")
    
    data_generator = DataGenerator(
        dataset_path=cfg.data.dataset_path,
        dataset_subsets=cfg.data.dataset_subsets,
        preference_ratio=cfg.data.preference_ratio,
        similarity_ratio=cfg.data.similarity_ratio,
        max_frames=cfg.data.max_frames,
        dataset_preference_ratio=cfg.data.dataset_preference_ratio,
        shuffle=cfg.data.shuffle,
        seed=cfg.data.seed,
        num_proc=cfg.data.num_proc,
        debug=cfg.debug,
        force_reprocess=cfg.data.force_reprocess
    )
    
    if rank == 0:
        rank_0_print(f"Data generator initialized on rank {rank}")
    
    return data_generator


def setup_eval_data_generator(cfg: ExperimentConfig) -> DataGenerator:
    """Shared function to create DataGenerator for evaluation"""
    
    # Get current rank for logging
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        rank_0_print(f"Setting up evaluation data generator on rank {rank}...")
    
    # Use eval-specific settings if provided, otherwise fall back to training settings
    eval_dataset_path = cfg.data.eval_dataset_path or cfg.data.dataset_path
    eval_dataset_subsets = cfg.data.eval_dataset_subsets or cfg.data.dataset_subsets
    
    eval_data_generator = DataGenerator(
        dataset_path=eval_dataset_path,
        dataset_subsets=eval_dataset_subsets,
        preference_ratio=cfg.data.preference_ratio,
        similarity_ratio=cfg.data.similarity_ratio,
        max_frames=cfg.data.max_frames,
        dataset_preference_ratio=cfg.data.dataset_preference_ratio,
        shuffle=cfg.data.shuffle,
        seed=cfg.data.seed + 1000,  # Different seed for eval to avoid overlap
        num_proc=cfg.data.num_proc,
        debug=cfg.debug,
        force_reprocess=False
    )
    
    if rank == 0:
        rank_0_print(f"Evaluation data generator initialized on rank {rank}")
        rank_0_print(f"  - Dataset path: {eval_dataset_path}")
        rank_0_print(f"  - Subsets: {eval_dataset_subsets}")
    
    return eval_data_generator


def setup_dataset(data_generator: DataGenerator, max_samples: int = 1000000, dataset_type: str = "train") -> InfiniteDataGeneratorDataset:
    """Shared function to create training or evaluation dataset"""
    
    rank_0_print(f"Setting up {dataset_type} dataset with max_samples={max_samples}")
    
    dataset = InfiniteDataGeneratorDataset(data_generator, max_samples=max_samples)
    
    rank_0_print(f"{dataset_type.capitalize()} dataset created successfully")
    return dataset


def setup_eval_dataset(cfg: ExperimentConfig) -> InfiniteDataGeneratorDataset:
    """Create evaluation dataset using eval-specific configuration"""
    
    # Create evaluation data generator
    eval_data_generator = setup_eval_data_generator(cfg)
    
    # Create evaluation dataset with limited size
    eval_dataset = setup_dataset(
        eval_data_generator, 
        max_samples=cfg.data.eval_subset_size,
        dataset_type="evaluation"
    )
    
    return eval_dataset


def setup_train_dataset(data_generator: DataGenerator, max_samples: int = 1000000) -> InfiniteDataGeneratorDataset:
    """Shared function to create training dataset (deprecated - use setup_dataset instead)"""
    
    rank_0_print(f"Setting up training dataset with max_samples={max_samples}")
    
    dataset = InfiniteDataGeneratorDataset(data_generator, max_samples=max_samples)
    
    rank_0_print("Training dataset created successfully")
    return dataset


def setup_batch_collator(processor: AutoProcessor, cfg: ExperimentConfig) -> BatchCollator:
    """Shared function to create BatchCollator"""
    
    rank_0_print("Setting up batch collator...")
    
    batch_collator = BatchCollator(
        processor=processor,
        max_length=cfg.training.max_seq_length,
        resized_height=cfg.data.resized_height,
        resized_width=cfg.data.resized_width
    )
    
    rank_0_print("Batch collator created successfully")
    return batch_collator 