#!/usr/bin/env python3
"""
Shared setup utilities for RFM training.
This file contains setup functions that can be reused across different training scripts.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    TrainingArguments,
)

from rfm.configs.experiment_configs import DataConfig, ExperimentConfig, ModelConfig
from rfm.data.collators import BaseCollator, ReWiNDBatchCollator, RFMBatchCollator, VQABatchCollator
from rfm.data.datasets import (
    ConfusionMatrixDataset,
    MixedDataset,
    PairedSuccessFailureDataset,
    ProgressDataset,
    RewardAlignmentDataset,
    RFMBaseDataset,
    WrongTaskDataset,
)
from rfm.models import RFM, RFMVQA, ReWiNDTransformer
from rfm.utils.logging import rank_0_print


def setup_model_and_processor(cfg: ModelConfig, hf_model_id: str = "") -> tuple[AutoProcessor, RFM]:
    """Shared function to set up model, processor, and tokenizer for both training and evaluation"""

    # Load processor and tokenizer
    if "SmolVLM" in cfg.base_model_id or "Qwen" in cfg.base_model_id:
        if "SmolVLM" in cfg.base_model_id:
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                padding_side="left",
                size={"longest_edge": 512},
                max_image_size={"longest_edge": 512}
            )
            
            rank_0_print(f"SmolVLM Processor: {processor}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch.bfloat16,
                # _attn_implementation="flash_attention_2",
            )
            rfm_model_cls = RFM  
        
    
        elif "Qwen" in cfg.base_model_id:
            if cfg.model_type == "default":
                qwen_model_cls = Qwen2_5_VLModel
                rfm_model_cls = RFM
            elif cfg.model_type == "vqa":
                qwen_model_cls = Qwen2_5_VLForConditionalGeneration
                rfm_model_cls = RFMVQA

            base_model = qwen_model_cls.from_pretrained(cfg.base_model_id)
            
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                # temporal_patch_size=1,
                # fps=1,
                # num_frames=cfg.data.max_frames,
                do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
                # max_frames=cfg.data.max_frames,
                padding_side="left",
                attn_implementation="flash_attention_2",
            )

            rank_0_print(f"Qwen Processor: {processor}")
        else:
            raise ValueError(f"Invalid base model id: {cfg.base_model_id}")
        
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token


        # Add RFM special tokens if they don't exist
        if cfg.model_type == "default":
            special_tokens = ["<|split_token|>", "<|reward_token|>", "<|pref_token|>"]
        else:
            special_tokens = []

        # Add special tokens to the tokenizer
        for token in special_tokens:
            if token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
                rank_0_print(f"Added special token: {token}")

        # Resize token embeddings if new tokens were added
        if len(processor.tokenizer) != base_model.config.vocab_size:
            rank_0_print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(processor.tokenizer)}")
            base_model.resize_token_embeddings(len(processor.tokenizer))
            rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")

        # Initialize RFM model wrapper with the pre-loaded base model
        rank_0_print("Initializing RFM model...")
        rfm_model = rfm_model_cls(config=base_model.config, processor=processor, base_model=base_model, base_model_id=cfg.base_model_id)

        if hf_model_id:
            rank_0_print(f"Loading model from {hf_model_id}")

            # before = rfm_model.model.visual.blocks[0].mlp.down_proj.weight
            # before = rfm_model.preference_head.weight
            # load the model from the evaluation path
            rfm_model = rfm_model_cls.from_pretrained(hf_model_id, processor=processor, base_model=base_model)

        tokenizer = processor.tokenizer

    elif "rewind_transformer" in cfg.base_model_id:
        # Pretrained image and text encoders
        image_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        train_img = cfg.train_vision_encoder
        train_text = cfg.train_language_model

        for p in image_encoder.parameters():
            p.requires_grad = train_img

        for p in text_encoder.parameters():
            p.requires_grad = train_text

        rank_0_print("Initializing ReWiND model...")
        rfm_model = ReWiNDTransformer(config=cfg, image_encoder=image_encoder, text_encoder=text_encoder)

    rank_0_print("Model architecture initialized")
    rank_0_print(f"Model architecture: {rfm_model}")

    return tokenizer, processor, rfm_model


def setup_peft_model(rfm_model: RFM, cfg: ExperimentConfig) -> RFM:
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
            param.requires_grad = True

    if cfg.logging.print_trainable_parameters:
        # Count trainable parameters manually - defer printing until after FSDP setup
        trainable_params = sum(p.numel() for p in rfm_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in rfm_model.parameters())
        rank_0_print(
            f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
        )
        rank_0_print("Training configuration:")
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


def setup_dataset(cfg: DataConfig, is_eval: bool = False, **kwargs) -> RFMBaseDataset:
    """Shared function to create DataGenerator for training or evaluation"""

    rank_0_print(f"Setting up data generator for {'evaluation' if is_eval else 'training'}...")

    if is_eval:
        datasets = cfg.eval_datasets
        subsets = cfg.eval_subsets
    else:
        datasets = cfg.train_datasets
        subsets = cfg.train_subsets

    # Validate that train_datasets and train_subsets have the same length
    if len(datasets) != len(subsets):
        raise ValueError(
            f"datasets and subsets must have the same length. Got {len(datasets)} datasets and {len(subsets)} subsets"
        )

    rank_0_print(f"Loading {len(datasets)} datasets with corresponding subsets")
    for i, (dataset, subset) in enumerate(zip(datasets, subsets, strict=False)):
        rank_0_print(f"  Dataset {i + 1}: {dataset} -> {subset}")

    dataset_cls = {
        "reward_alignment": RewardAlignmentDataset,
        "success_failure": PairedSuccessFailureDataset,
        "policy_ranking": ProgressDataset,
        "confusion_matrix": ConfusionMatrixDataset,
        "wrong_task": WrongTaskDataset,
        "default": MixedDataset,
    }

    dataset = dataset_cls[cfg.dataset_type](config=cfg, is_evaluation=is_eval, **kwargs)
    rank_0_print("Dataset initialized")
    return dataset


def setup_batch_collator(processor: AutoProcessor, tokenizer: AutoTokenizer, cfg: ExperimentConfig) -> BaseCollator:
    """Shared function to create BatchCollator"""

    rank_0_print("Setting up batch collator...")
    collator_kwargs = {
        "processor": processor,
        "max_length": cfg.training.max_seq_length,
        "resized_height": cfg.data.resized_height,
        "resized_width": cfg.data.resized_width,
        "base_model_id": cfg.model.base_model_id,
    }
    if "Qwen" in cfg.model.base_model_id or "SmolVLM" in cfg.model.base_model_id:
        if cfg.model.model_type == "default":
            batch_collator = RFMBatchCollator(**collator_kwargs)
        elif cfg.model.model_type == "vqa":
            batch_collator = VQABatchCollator(**collator_kwargs, inference=cfg.mode == "eval")
    elif "rewind_transformer" in cfg.model.base_model_id:
        batch_collator = ReWiNDBatchCollator(**collator_kwargs, tokenizer=tokenizer)

    rank_0_print("Batch collator created successfully")
    return batch_collator
