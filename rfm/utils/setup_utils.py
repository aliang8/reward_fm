#!/usr/bin/env python3
"""
Shared setup utilities for RFM training.
This file contains setup functions that can be reused across different training scripts.
"""

from unsloth import FastVisionModel

import re
import os
from pathlib import Path
from typing import Tuple, Optional, Any
import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from huggingface_hub import HfApi
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    TrainingArguments,
    BitsAndBytesConfig,
)

# Try to import Qwen3 models if available
try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel

    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False
    Qwen3VLForConditionalGeneration = None
    Qwen3VLModel = None

from rfm.configs.experiment_configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    TrainingConfig,
)
from rfm.data.collators import BaseCollator, ReWiNDBatchCollator, RFMBatchCollator, VQABatchCollator
from rfm.data.datasets import (
    RFMDataset,
    StrategyBalancedDataset,
    BaseDataset,
    RepeatedDataset,
    SingleFrameDataset,
)
from rfm.data.datasets.custom_eval import CustomEvalDataset
from rfm.data.datasets.data_source_balance import DataSourceBalancedWrapper
from rfm.models import RFM, RFMVQA, ReWiNDTransformer
from rfm.models.rewind_transformer import ReWINDTransformerConfig
from rfm.models.rewind_transformer_scale import ReWINDScaleTransformerConfig, ReWiNDScaleTransformer
from rfm.utils.logger import get_logger

logger = get_logger()
from rfm.utils.save import parse_hf_model_id_and_revision, resolve_checkpoint_path


def _load_checkpoint_weights_from_safetensors(model, checkpoint_path: str) -> None:
    """
    Load checkpoint weights from safetensors files in a checkpoint directory.

    This is needed when using Unsloth, as we can't use from_pretrained on checkpoints.
    Instead, we load the base model with Unsloth first, then manually load the checkpoint weights.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint directory containing safetensors files
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")

    # Collect all safetensors files
    safetensors_files = list(checkpoint_path.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in checkpoint directory: {checkpoint_path}")

    logger.info(f"Loading checkpoint weights from {len(safetensors_files)} safetensors file(s) in {checkpoint_path}")

    # Load all safetensors files and merge into a single state dict
    state_dict = {}
    for safetensors_file in safetensors_files:
        logger.debug(f"Loading weights from {safetensors_file.name}")
        file_state_dict = load_file(str(safetensors_file))
        state_dict.update(file_state_dict)

    # Load state dict into model with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
        logger.debug(
            f"Missing keys: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}"
        )

    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
        logger.debug(
            f"Unexpected keys: {unexpected_keys[:10]}..."
            if len(unexpected_keys) > 10
            else f"Unexpected keys: {unexpected_keys}"
        )

    logger.info(f"Successfully loaded checkpoint weights from {checkpoint_path}")


def _load_base_model_with_unsloth(
    cfg: ModelConfig,
    torch_dtype: torch.dtype,
    extra_kwargs: dict,
    peft_config: Optional[PEFTConfig] = None,
    loading_from_checkpoint: bool = False,
) -> Tuple[Any, Any]:
    """
    Load base model using Unsloth's FastVisionModel.

    Args:
        cfg: Model configuration
        torch_dtype: Torch dtype to use
        extra_kwargs: Extra kwargs for model loading (e.g., attn_implementation)
        peft_config: Optional PEFT configuration
        loading_from_checkpoint: If True, skip PEFT application (checkpoint already has weights)

    Returns:
        Tuple of (base_model, tokenizer)
    """
    logger.info("Using Unsloth for faster training with Qwen model")

    # Load model with unsloth
    base_model, tokenizer = FastVisionModel.from_pretrained(
        cfg.base_model_id,
        load_in_4bit=cfg.quantization,  # Use 4bit if quantization is enabled
        use_gradient_checkpointing="unsloth",  # Use unsloth's optimized checkpointing
        dtype=torch_dtype,  # Set the dtype from config,
        full_finetuning=True,
        device_map=None,
        attn_implementation=extra_kwargs["attn_implementation"],
        trust_remote_code=True,
    )
    if cfg.model_type == "default":
        base_model = base_model.model

    # Apply PEFT if enabled (only if not loading from checkpoint)
    # When loading from checkpoint, the checkpoint already contains the trained weights
    if cfg.use_peft and peft_config and not loading_from_checkpoint:
        logger.info("Applying PEFT configuration to base model")
        base_model = FastVisionModel.get_peft_model(
            base_model,
            finetune_vision_layers=cfg.train_vision_encoder,
            finetune_language_layers=cfg.train_language_model,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
        )
    elif loading_from_checkpoint:
        logger.info("Skipping PEFT application - loading from checkpoint which already contains trained weights")

    return base_model, tokenizer


def _load_base_model_standard(
    cfg: ModelConfig,
    torch_dtype: torch.dtype,
    extra_kwargs: dict,
    bnb: Optional[BitsAndBytesConfig],
) -> Any:
    """
    Load base model using standard transformers loading.

    Args:
        cfg: Model configuration
        torch_dtype: Torch dtype to use
        extra_kwargs: Extra kwargs for model loading (e.g., attn_implementation)
        bnb: Optional BitsAndBytesConfig for quantization

    Returns:
        Base model
    """
    # Check if it's Molmo, Qwen3 or Qwen2/2.5
    is_molmo = "Molmo" in cfg.base_model_id
    is_qwen3 = ("Qwen3" in cfg.base_model_id or "qwen3" in cfg.base_model_id.lower()) and HAS_QWEN3

    # Select appropriate model classes based on version and model type
    if is_molmo:
        # Molmo2 uses AutoModelForImageTextToText with trust_remote_code
        base_model = AutoModelForImageTextToText.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
            **extra_kwargs,
            quantization_config=bnb,
        )
        if cfg.model_type == "default":
            # For RFM (non-VQA), extract the base model
            base_model = base_model.model
        logger.info("Using Molmo2 models")
    elif is_qwen3:
        qwen_model_cls = Qwen3VLModel if cfg.model_type == "default" else Qwen3VLForConditionalGeneration
        base_model = qwen_model_cls.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            **extra_kwargs,
            quantization_config=bnb,
        )
        logger.info("Using Qwen3 models")
    else:
        qwen_model_cls = Qwen2_5_VLModel if cfg.model_type == "default" else Qwen2_5_VLForConditionalGeneration
        base_model = qwen_model_cls.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            **extra_kwargs,
            quantization_config=bnb,
        )
        logger.info("Using Qwen2/2.5 models")

    return base_model


def _setup_processor_and_tokenizer(cfg: ModelConfig) -> AutoProcessor:
    """
    Setup processor and tokenizer for the model.

    Args:
        cfg: Model configuration

    Returns:
        Processor
    """
    if "SmolVLM" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
            size={"longest_edge": 512},
            max_image_size={"longest_edge": 512},
            use_fast=True,
        )
        logger.info(f"SmolVLM Processor: {processor}")
    elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
            padding_side="left",
        )
        logger.info(f"Qwen Processor: {processor}")
    else:
        raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor


def _add_special_tokens_and_resize(cfg: ModelConfig, processor: AutoProcessor, base_model: Any) -> None:
    """
    Add RFM special tokens and resize token embeddings if needed.

    Args:
        cfg: Model configuration
        processor: Processor with tokenizer
        base_model: Base model to resize embeddings for
    """
    # Add RFM special tokens if they don't exist
    if cfg.model_type == "default":
        special_tokens = [
            "<|split_token|>",
            "<|reward_token|>",
            "<|pref_token|>",
            "<|sim_token|>",
            "<|prog_token_A|>",
            "<|prog_token_B|>",
            "<|succ_token_A|>",
            "<|succ_token_B|>",
        ]
    else:
        special_tokens = []

    # Add special tokens to the tokenizer
    if cfg.model_type != "vqa":
        for token in special_tokens:
            if token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
                logger.info(f"Added special token: {token}")

    # Resize token embeddings if new tokens were added
    vocab_size = (
        base_model.config.text_config.vocab_size
        if ("Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id)
        else base_model.config.vocab_size
    )

    if len(processor.tokenizer) != vocab_size:
        logger.info(f"Resizing token embeddings from {vocab_size} to {len(processor.tokenizer)}")

        is_molmo = "Molmo" in cfg.base_model_id
        if is_molmo:
            # Custom resize for Molmo2 - its Molmo2Embedding stores embedding as a Parameter directly
            new_vocab_size = len(processor.tokenizer)
            _embed_layer = base_model.get_input_embeddings()

            # Check if embedding is a Parameter (tensor) directly, or an nn.Embedding
            if hasattr(_embed_layer, "embedding"):
                old_embed_attr = _embed_layer.embedding

                # Case 1: embedding is a Parameter (raw tensor)
                if isinstance(old_embed_attr, torch.nn.Parameter):
                    old_num_tokens, embedding_dim = old_embed_attr.shape

                    # Create new parameter with expanded vocab
                    new_embed_data = torch.zeros(
                        new_vocab_size, embedding_dim, device=old_embed_attr.device, dtype=old_embed_attr.dtype
                    )

                    # Copy existing weights
                    new_embed_data[:old_num_tokens] = old_embed_attr.data

                    # Initialize new token embeddings using mean of existing embeddings
                    mean_embedding = old_embed_attr.data.mean(dim=0)
                    new_embed_data[old_num_tokens:] = mean_embedding.unsqueeze(0).expand(
                        new_vocab_size - old_num_tokens, -1
                    )

                    # Replace the embedding Parameter
                    _embed_layer.embedding = torch.nn.Parameter(new_embed_data)

                    # Also update config to reflect new vocab size
                    base_model.config.text_config.vocab_size = new_vocab_size

                    logger.info(
                        f"Custom resized Molmo2 embeddings (Parameter) from {old_num_tokens} to {new_vocab_size}"
                    )

                # Case 2: embedding is an nn.Embedding with .weight
                elif hasattr(old_embed_attr, "weight"):
                    old_num_tokens, embedding_dim = old_embed_attr.weight.shape

                    # Create new embedding layer with expanded vocab
                    new_embedding = torch.nn.Embedding(
                        new_vocab_size,
                        embedding_dim,
                        device=old_embed_attr.weight.device,
                        dtype=old_embed_attr.weight.dtype,
                    )

                    # Copy existing weights
                    new_embedding.weight.data[:old_num_tokens] = old_embed_attr.weight.data

                    # Initialize new token embeddings using mean of existing embeddings
                    mean_embedding = old_embed_attr.weight.data.mean(dim=0)
                    new_embedding.weight.data[old_num_tokens:] = mean_embedding.unsqueeze(0).expand(
                        new_vocab_size - old_num_tokens, -1
                    )

                    # Replace the nested embedding
                    _embed_layer.embedding = new_embedding

                    # Also update config to reflect new vocab size
                    base_model.config.text_config.vocab_size = new_vocab_size

                    logger.info(
                        f"Custom resized Molmo2 embeddings (Embedding) from {old_num_tokens} to {new_vocab_size}"
                    )
                else:
                    logger.warning(f"Cannot resize Molmo2 embeddings - unknown embedding type: {type(old_embed_attr)}")
            else:
                logger.warning(f"Cannot resize Molmo2 embeddings - no embedding attribute found")
        else:
            base_model.resize_token_embeddings(len(processor.tokenizer))
            logger.info(f"Resized token embeddings to {len(processor.tokenizer)}")


def _verify_checkpoint_loading(cfg: ModelConfig, model: Any, before_weights: dict) -> None:
    """
    Verify that checkpoint weights were loaded correctly by comparing before/after weights.

    Args:
        cfg: Model configuration
        model: The model after loading checkpoint
        before_weights: Dictionary of weights before loading (keys: visual, progress_head, lm_embed_tokens, lm_layer)
    """
    if cfg.model_type == "vqa":
        return

    if "Qwen2.5" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.down_proj.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.linear_fc1.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    else:
        return

    before_visual = before_weights["visual"]
    before_progress_head = before_weights["progress_head"]
    before_lm_embed_tokens = before_weights["lm_embed_tokens"]
    before_lm_layer = before_weights["lm_layer"]

    logger.info(
        f"Before visual: {before_visual.shape}, {before_visual.sum()} | After visual: {after_visual.shape}, {after_visual.sum()}"
    )
    logger.info(
        f"Before progress head: {before_progress_head.shape}, {before_progress_head.sum()} | After progress head: {after_progress_head.shape}, {after_progress_head.sum()}"
    )
    logger.info(
        f"Before LM embed tokens: {before_lm_embed_tokens.shape}, {before_lm_embed_tokens.sum()} | After LM embed tokens: {after_lm_embed_tokens.shape}, {after_lm_embed_tokens.sum()}"
    )
    logger.info(
        f"Before LM layer: {before_lm_layer.shape}, {before_lm_layer.sum()} | After LM layer: {after_lm_layer.shape}, {after_lm_layer.sum()}"
    )

    # check that before and after are different
    if torch.allclose(before_visual, after_visual):
        logger.warning("Before and after visual are the same! Check if you loaded the pretrained model correctly")
    if torch.allclose(before_progress_head, after_progress_head):
        logger.warning(
            "Before and after progress head are the same! Check if you loaded the pretrained model correctly"
        )
    if torch.allclose(before_lm_embed_tokens, after_lm_embed_tokens):
        logger.warning(
            "Before and after LM embed tokens are the same! Check if you loaded the pretrained model correctly"
        )


def setup_model_and_processor(
    cfg: ModelConfig, hf_model_id: str = "", peft_config: PEFTConfig = None
) -> tuple[AutoProcessor, RFM]:
    """
    Shared function to set up model, processor, and tokenizer for both training and evaluation.

    Args:
        cfg: Model configuration
        hf_model_id: Optional HuggingFace model ID to load from

    Note:
        When use_unsloth is enabled for Qwen models:
        - The model will be loaded using unsloth's FastVisionModel
        - Automatically uses optimized gradient checkpointing
        - If use_peft is enabled, applies unsloth's optimized PEFT configuration
        - Use unsloth/Qwen models for best performance (e.g., unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit)
    """

    # Convert string dtype to torch dtype (used across all model loading paths)
    torch_dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
    logger.info(f"Using torch dtype: {torch_dtype}")

    # Check if unsloth should be used
    use_unsloth = cfg.use_unsloth and "Qwen" in cfg.base_model_id

    if use_unsloth:
        logger.info("Unsloth mode enabled for faster training")

    # If quantization is enabled, use bitsandbytes (unless using unsloth)
    if cfg.quantization and not use_unsloth:
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb = None

    try:
        import flash_attn

        logger.info("Flash Attention 2 CUDA is available")
        has_flash_attn = True
    except:
        logger.info("Flash Attention 2 CUDA is not available")
        has_flash_attn = False

    if has_flash_attn:
        extra_kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        extra_kwargs = {"attn_implementation": "sdpa"}

    # Determine if we're loading from a checkpoint
    loading_from_checkpoint = bool(hf_model_id)

    # Load processor and tokenizer
    if "SmolVLM" in cfg.base_model_id or "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        if "SmolVLM" in cfg.base_model_id:
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                padding_side="left",
                size={"longest_edge": 512},
                max_image_size={"longest_edge": 512},
                use_fast=True,
            )
            logger.info(f"SmolVLM Processor: {processor}")

            base_model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                **extra_kwargs,
                quantization_config=bnb,
            )
            model_cls = RFM if cfg.model_type == "default" else RFMVQA

        elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
            # Load base model (with or without Unsloth)
            if use_unsloth:
                base_model, tokenizer = _load_base_model_with_unsloth(
                    cfg, torch_dtype, extra_kwargs, peft_config, loading_from_checkpoint=loading_from_checkpoint
                )
            else:
                base_model = _load_base_model_standard(cfg, torch_dtype, extra_kwargs, bnb)
                tokenizer = None  # Will be loaded with processor

            model_cls = RFM if cfg.model_type == "default" else RFMVQA

            # Setup processor and tokenizer
            processor = _setup_processor_and_tokenizer(cfg)
            if tokenizer is None:
                tokenizer = processor.tokenizer

        else:
            raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

        # Add special tokens and resize embeddings
        _add_special_tokens_and_resize(cfg, processor, base_model)

        # Initialize RFM model wrapper with the pre-loaded base model
        logger.info("Initializing RFM model...")
        tokenizer = processor.tokenizer

        model = model_cls(
            config=base_model.config,
            processor=processor,
            tokenizer=tokenizer,
            base_model=base_model,
            base_model_id=cfg.base_model_id,
            model_config=cfg,  # Pass ModelConfig for RFM-specific settings
        )

        # Load checkpoint if provided
        if hf_model_id:
            repo_id, revision_to_load = parse_hf_model_id_and_revision(hf_model_id, model_name="model")

            # Capture before weights for verification
            before_weights = {}
            if cfg.model_type != "vqa":
                if "Qwen2.5" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.down_proj.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }
                elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.linear_fc1.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }

            # Load the model from the evaluation path
            model = model_cls.from_pretrained(
                repo_id,
                processor=processor,
                tokenizer=tokenizer,
                base_model=base_model,
                base_model_id=cfg.base_model_id,
                model_config=cfg,
                revision=revision_to_load,
            )

            # Verify weights were loaded
            if before_weights:
                _verify_checkpoint_loading(cfg, model, before_weights)

    # elif "rewind_transformer" in cfg.base_model_id or "rewind_scale_transformer" in cfg.base_model_id:
    elif "rewind" in cfg.base_model_id:
        # Initialize new model with encoders
        # Pretrained image and text encoders
        image_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        if hf_model_id:
            repo_id, revision_to_load = parse_hf_model_id_and_revision(hf_model_id, model_name="ReWiND model")

            model = ReWiNDTransformer.from_pretrained(
                repo_id,
                processor=processor,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                revision=revision_to_load,
            )
        else:
            train_img = cfg.train_vision_encoder
            train_text = cfg.train_language_model

            for p in image_encoder.parameters():
                p.requires_grad = train_img

            for p in text_encoder.parameters():
                p.requires_grad = train_text

            logger.info("Initializing ReWiND model...")

            rewind_config = cfg.rewind if cfg.rewind is not None else ReWINDTransformerConfig()
            if cfg.rewind_scale_model:
                rewind_config = ReWINDScaleTransformerConfig(causal_mask=cfg.causal_mask)
                model = ReWiNDScaleTransformer(
                    config=rewind_config,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                )
            else:
                model = ReWiNDTransformer(
                    config=rewind_config,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                )

    logger.info("Model architecture initialized")
    logger.info(f"Model architecture: {model}")

    # Configure which parts of the model to train based on config
    for name, param in model.named_parameters():
        # Train prediction heads based on individual settings
        if "progress_head" in name:
            param.requires_grad = cfg.train_progress_head
        elif "success_head" in name:
            param.requires_grad = cfg.train_success_head
        elif "preference_head" in name:
            param.requires_grad = cfg.train_preference_head
        elif "similarity_head" in name:
            param.requires_grad = cfg.train_similarity_head
        # Train vision encoder if specified
        elif "visual" in name or "vision" in name:
            # if PEFT enabled, we don't need to do anything
            if cfg.use_peft and cfg.peft_vision_encoder:
                pass
            else:
                param.requires_grad = cfg.train_vision_encoder
        elif "language_model" in name:
            param.requires_grad = cfg.train_language_model
        elif "text_encoder" in name:
            param.requires_grad = cfg.train_language_model
        elif "text_model" in name:
            param.requires_grad = cfg.train_language_model
        elif "image_encoder" in name:
            param.requires_grad = cfg.train_vision_encoder
        else:
            param.requires_grad = True

        if "SmolVLM" in cfg.base_model_id:
            if "text_model" in name:
                param.requires_grad = cfg.train_language_model
            if "vision_model" in name:
                param.requires_grad = cfg.train_vision_encoder
            # i think we want to train the connector head always
            # we don't need the lm_head to be trainable
            if "lm_head" in name:
                param.requires_grad = False

    logger.info("Training configuration:")
    logger.info(f"  - Vision encoder: {cfg.train_vision_encoder}")
    logger.info(f"  - Language model: {cfg.train_language_model}")
    logger.info(f"  - Progress head: {cfg.train_progress_head}")
    logger.info(f"  - Success head: {getattr(cfg, 'train_success_head', False)}")
    logger.info(f"  - Preference head: {cfg.train_preference_head}")
    logger.info(f"  - Similarity head: {cfg.train_similarity_head}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name:60} | {param.shape} | RG: {param.requires_grad}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return tokenizer, processor, model


def setup_peft_model(rfm_model: RFM, cfg: PEFTConfig) -> RFM:
    """Shared function to apply PEFT configuration to the model"""

    logger.info("Using PEFT/LoRA training...")
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
    )
    if cfg.peft_vision_encoder:
        # vision backbone is frozen, but we can still train the LoRA parameters
        logger.info("Attaching LoRA to only the vision encoder...")
        rfm_model.base_model.model.visual = get_peft_model(rfm_model.base_model.model.visual, lora_config)

    # Count trainable parameters manually - defer printing until after FSDP setup
    trainable_params = sum(p.numel() for p in rfm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in rfm_model.parameters())
    logger.info(
        f"AFTER PEFT: trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return rfm_model


def create_training_arguments(cfg: TrainingConfig, output_dir: str, is_eval: bool = False) -> TrainingArguments:
    """Shared function to create TrainingArguments for both training and evaluation"""

    # Base arguments that are the same for both training and evaluation
    base_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
        "learning_rate": cfg.learning_rate,
        "save_strategy": cfg.save_strategy,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "remove_unused_columns": cfg.remove_unused_columns,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataloader_pin_memory": cfg.dataloader_pin_memory,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "dataloader_persistent_workers": cfg.dataloader_persistent_workers,
        "save_safetensors": True,
        "save_total_limit": 2,
        # Evaluation settings
        "eval_strategy": cfg.evaluation_strategy,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "do_eval": cfg.do_eval,
        "prediction_loss_only": cfg.prediction_loss_only,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_steps": cfg.warmup_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "max_grad_norm": cfg.max_grad_norm,
        "weight_decay": cfg.weight_decay,
        "disable_tqdm": False,
        # # Compile settings
        # "torch_compile": True,
        # "torch_compile_mode": "max-autotune",
        # "torch_compile_backend": "inductor",
    }

    # Add eval_steps if evaluation_strategy is "steps"
    if cfg.evaluation_strategy == "steps" and cfg.eval_steps is not None:
        base_args["eval_steps"] = cfg.eval_steps

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
            "num_train_epochs": cfg.num_train_epochs if cfg.num_train_epochs is not None else 1,
            "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
            # Disable HuggingFace's automatic logging - we use custom Logger class instead
            "report_to": "none",
        })

    return TrainingArguments(**base_args)


def setup_dataset(cfg: DataConfig, is_eval: bool = False, **kwargs) -> BaseDataset:
    """Shared function to create Dataset for training or evaluation"""
    dataset_cls = {
        "rfm": RFMDataset,
        "strategy_balance": StrategyBalancedDataset,
        "single_frame": SingleFrameDataset,
    }
    
    # Validate dataset_type
    if cfg.dataset_type not in dataset_cls:
        raise ValueError(
            f"Unknown dataset_type: {cfg.dataset_type}. "
            f"Must be one of: {list(dataset_cls.keys())}"
        )
    
    # Create the base dataset
    dataset = dataset_cls[cfg.dataset_type](config=cfg, is_evaluation=is_eval, **kwargs)
    
    # Apply data source balancing wrapper if requested
    if cfg.use_data_source_balance:
        if not cfg.data_source_weights:
            raise ValueError(
                "use_data_source_balance=True requires data_source_weights to be set in config"
            )
        dataset = DataSourceBalancedWrapper(dataset, config=cfg, is_evaluation=is_eval, **kwargs)

    if not is_eval:
        dataset = RepeatedDataset(dataset)
    return dataset


def setup_custom_eval_dataset(
    cfg: DataConfig, sampler_type: str, is_eval: bool = False, verbose=True, sampler_kwargs=None
):
    """Setup a custom evaluation dataset with a specific sampler."""
    custom_eval_dataset = CustomEvalDataset(
        sampler_type, cfg, is_evaluation=is_eval, verbose=verbose, sampler_kwargs=sampler_kwargs
    )

    return custom_eval_dataset


def setup_batch_collator(
    processor: AutoProcessor, tokenizer: AutoTokenizer, cfg: ExperimentConfig, is_eval: bool = False
) -> BaseCollator:
    """Shared function to create BatchCollator"""
    collator_kwargs = {
        "processor": processor,
        "resized_height": cfg.data.resized_height,
        "resized_width": cfg.data.resized_width,
        "base_model_id": cfg.model.base_model_id,
        "use_multi_image": cfg.data.use_multi_image,
        "prog_pref": cfg.training.predict_pref_progress,
        "prog_sim": cfg.training.predict_sim_progress,
        "use_progress_token": cfg.model.use_progress_token,
        "shuffle_progress_frames": cfg.data.shuffle_progress_frames,
        "inference": is_eval,
    }
    # Check for unsupported Molmo2 video mode
    if "Molmo" in cfg.model.base_model_id and not cfg.data.use_multi_image:
        raise ValueError(
            "Molmo2 implementation does not yet support video mode as it requires extra imports (use_multi_image=False). "
            "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
        )

    if "Qwen" in cfg.model.base_model_id or "SmolVLM" in cfg.model.base_model_id or "Molmo" in cfg.model.base_model_id:
        if cfg.model.model_type == "default":
            batch_collator = RFMBatchCollator(**collator_kwargs)
        elif cfg.model.model_type == "vqa":
            batch_collator = VQABatchCollator(**collator_kwargs)
    # elif "rewind_transformer" in cfg.model.base_model_id:
    elif "rewind" in cfg.model.base_model_id:
        batch_collator = ReWiNDBatchCollator(
            **collator_kwargs, tokenizer=tokenizer, load_embeddings=cfg.data.load_embeddings
        )
    return batch_collator
