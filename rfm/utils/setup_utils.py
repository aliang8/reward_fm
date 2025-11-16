#!/usr/bin/env python3
"""
Shared setup utilities for RFM training.
This file contains setup functions that can be reused across different training scripts.
"""

from unsloth import FastVisionModel

import re
from typing import Tuple, Optional
import torch
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

from rfm.configs.experiment_configs import DataConfig, ExperimentConfig, ModelConfig, PEFTConfig
from rfm.data.collators import BaseCollator, ReWiNDBatchCollator, RFMBatchCollator, VQABatchCollator
from rfm.data.datasets import (
    BalancedRFMDataset,
    RFMDataset,
    BaseDataset,
    InfiniteDataset,
)
from rfm.data.datasets.custom_eval import CustomEvalDataset
from rfm.models import RFM, RFMVQA, ReWiNDTransformer
from rfm.models.rewind_transformer import ReWINDTransformerConfig
from rfm.utils.distributed import rank_0_print


def find_best_model_tag(hf_model_id: str, hub_token: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the best model tag from HuggingFace Hub by parsing tag names and extracting scores.

    Expected tag format: "best-{metric_short}-{score:.4f}-step-{step}"
    Example: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"

    Args:
        hf_model_id: HuggingFace model ID (e.g., "aliangdw/rewind-debug")
        hub_token: Optional HuggingFace token for private repos

    Returns:
        tuple: (best_tag_name, best_score) or (None, None) if no valid tags found
    """
    try:
        api = HfApi(token=hub_token)

        # Check if repository exists
        if not api.repo_exists(repo_id=hf_model_id, repo_type="model"):
            rank_0_print(f"Repository {hf_model_id} does not exist")
            return None, None

        # Get all tags for the repository
        tags = api.list_repo_refs(repo_id=hf_model_id, repo_type="model").tags

        if not tags:
            rank_0_print(f"No tags found in repository {hf_model_id}")
            return None, None

        rank_0_print(f"Found {len(tags)} tags in {hf_model_id}: {[tag.name for tag in tags]}")

        best_tag = None
        best_score = float("-inf")

        # Parse each tag to extract score
        for tag in tags:
            tag_name = tag.name

            # Match our tag pattern: "best-{metric_short}-{score}-step-{step}"
            # Examples: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"
            # Score can be positive or negative (e.g., 0.8500 or -1.2300)
            pattern = r"best-.*?-(-?\d+\.\d+)-step-\d+"
            match = re.search(pattern, tag_name)

            if match:
                try:
                    score = float(match.group(1))
                    rank_0_print(f"Parsed tag '{tag_name}': score = {score}")

                    if score > best_score:
                        best_score = score
                        best_tag = tag_name

                except ValueError:
                    rank_0_print(f"Could not parse score from tag '{tag_name}'")
                    continue
            else:
                rank_0_print(f"Tag '{tag_name}' does not match expected pattern")

        if best_tag:
            rank_0_print(f"Best tag found: '{best_tag}' with score {best_score}")
        else:
            rank_0_print("No valid tags found matching the expected pattern")

        return best_tag, best_score

    except Exception as e:
        rank_0_print(f"Error finding best tag for {hf_model_id}: {e}")
        return None, None


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
    rank_0_print(f"Using torch dtype: {torch_dtype}")

    # Check if unsloth should be used
    use_unsloth = cfg.use_unsloth and "Qwen" in cfg.base_model_id

    if use_unsloth:
        rank_0_print("Unsloth mode enabled for faster training")

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

        rank_0_print("Flash Attention 2 CUDA is available")
        has_flash_attn = True
    except:
        rank_0_print("Flash Attention 2 CUDA is not available")
        has_flash_attn = False

    if has_flash_attn:
        extra_kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        extra_kwargs = {"attn_implementation": "sdpa"}

    # Load processor and tokenizer
    if "SmolVLM" in cfg.base_model_id or "Qwen" in cfg.base_model_id:
        if "SmolVLM" in cfg.base_model_id:
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                # padding_side="left",
                size={"longest_edge": 512},
                max_image_size={"longest_edge": 512},
                use_fast=True,
            )

            rank_0_print(f"SmolVLM Processor: {processor}")

            base_model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                **extra_kwargs,
                quantization_config=bnb,
            )

            model_cls = RFM if cfg.model_type == "default" else RFMVQA

        elif "Qwen" in cfg.base_model_id:
            # Check if unsloth should be used
            if use_unsloth:
                rank_0_print("Using Unsloth for faster training with Qwen model")

                # Load model with unsloth
                base_model, tokenizer = FastVisionModel.from_pretrained(
                    cfg.base_model_id,
                    load_in_4bit=cfg.quantization,  # Use 4bit if quantization is enabled
                    use_gradient_checkpointing="unsloth",  # Use unsloth's optimized checkpointing
                    dtype=torch_dtype,  # Set the dtype from config,
                    full_finetuning=True,
                    device_map=None,
                    attn_implementation="sdpa",
                )
                if cfg.model_type == "default":
                    base_model = base_model.model

                # Apply PEFT if enabled
                if cfg.use_peft and peft_config:
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
            else:
                # Check if it's Qwen3 or Qwen2/2.5
                is_qwen3 = ("Qwen3" in cfg.base_model_id or "qwen3" in cfg.base_model_id.lower()) and HAS_QWEN3

                # Select appropriate model classes based on version and model type
                if is_qwen3:
                    qwen_model_cls = Qwen3VLModel if cfg.model_type == "default" else Qwen3VLForConditionalGeneration
                    rank_0_print("Using Qwen3 models")
                else:
                    qwen_model_cls = (
                        Qwen2_5_VLModel if cfg.model_type == "default" else Qwen2_5_VLForConditionalGeneration
                    )
                    rank_0_print("Using Qwen2/2.5 models")

                base_model = qwen_model_cls.from_pretrained(
                    cfg.base_model_id,
                    torch_dtype=torch_dtype,
                    **extra_kwargs,
                    quantization_config=bnb,
                )

            model_cls = RFM if cfg.model_type == "default" else RFMVQA

            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                # temporal_patch_size=1,
                # fps=1,
                # num_frames=cfg.data.max_frames,
                do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
                # max_frames=cfg.data.max_frames,
                padding_side="left",
            )
            rank_0_print(f"Qwen Processor: {processor}")
        else:
            raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

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
                    rank_0_print(f"Added special token: {token}")

            # Resize token embeddings if new tokens were added
            if len(processor.tokenizer) != base_model.config.vocab_size:
                rank_0_print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(processor.tokenizer)}")
                base_model.resize_token_embeddings(len(processor.tokenizer))
                rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")

        # Initialize RFM model wrapper with the pre-loaded base model
        rank_0_print("Initializing RFM model...")
        tokenizer = processor.tokenizer

        model = model_cls(
            config=base_model.config,
            processor=processor,
            tokenizer=tokenizer,
            base_model=base_model,
            base_model_id=cfg.base_model_id,
            model_config=cfg,  # Pass ModelConfig for RFM-specific settings
        )

        if hf_model_id:
            # Allow users to specify explicit revisions via repo@revision
            if "@" in hf_model_id:
                repo_id, explicit_revision = hf_model_id.split("@", 1)
            else:
                repo_id, explicit_revision = hf_model_id, None

            revision_to_load = explicit_revision

            # Check if this is a HuggingFace repo (not a local path) and find best tag
            if "/" in repo_id and not repo_id.startswith("/"):
                if revision_to_load:
                    rank_0_print(f"Loading model {repo_id} at explicit revision '{revision_to_load}'")
                else:
                    best_tag, best_score = find_best_model_tag(repo_id)
                    if best_tag:
                        revision_to_load = best_tag
                        rank_0_print(f"Loading model from best tag: {repo_id}@{revision_to_load} (score: {best_score})")
                    else:
                        rank_0_print(f"No best tag found, loading latest revision of {repo_id}")
                hf_repo_with_rev = repo_id
            else:
                hf_repo_with_rev = repo_id
                rank_0_print(f"Loading local/explicit model from {hf_repo_with_rev}")
            if cfg.model_type != "vqa":
                before = model.model.visual.blocks[0].mlp.down_proj.weight
                before = model.preference_head[0].weight
            # load the model from the evaluation path
            model = model_cls.from_pretrained(
                hf_repo_with_rev,
                processor=processor,
                tokenizer=tokenizer,
                base_model=base_model,
                base_model_id=cfg.base_model_id,
                model_config=cfg,
                revision=revision_to_load,
            )
            if cfg.model_type != "vqa":
                after = model.model.visual.blocks[0].mlp.down_proj.weight
                after = model.preference_head[0].weight
                rank_0_print(f"Before: {before.shape}, {before.sum()} | After: {after.shape}, {after.sum()}")
                # check that before and after are different
                if torch.allclose(before, after):
                    rank_0_print("Before and after are the same! Check if you loaded the pretrained model correctly")
                    import ipdb

                    ipdb.set_trace()

    elif "rewind_transformer" in cfg.base_model_id:
        # Initialize new model with encoders
        # Pretrained image and text encoders
        image_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        if hf_model_id:
            # Check if this is a HuggingFace repo (not a local path) and find best tag
            if "/" in hf_model_id and not hf_model_id.startswith("/") and not "@" in hf_model_id:
                best_tag, best_score = find_best_model_tag(hf_model_id)
                if best_tag:
                    rank_0_print(f"Loading ReWiND model from best tag: {hf_model_id} (score: {best_score})")
            else:
                rank_0_print(f"Loading ReWiND model from {hf_model_id}")

            model = ReWiNDTransformer.from_pretrained(
                hf_model_id,
                processor=processor,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                revision=best_tag,
            )
        else:
            train_img = cfg.train_vision_encoder
            train_text = cfg.train_language_model

            for p in image_encoder.parameters():
                p.requires_grad = train_img

            for p in text_encoder.parameters():
                p.requires_grad = train_text

            rank_0_print("Initializing ReWiND model...")
            rewind_config = cfg.rewind if cfg.rewind is not None else ReWINDTransformerConfig()
            model = ReWiNDTransformer(
                config=rewind_config,
                processor=processor,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
            )

    rank_0_print("Model architecture initialized")
    rank_0_print(f"Model architecture: {model}")

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

    rank_0_print("Training configuration:")
    rank_0_print(f"  - Vision encoder: {cfg.train_vision_encoder}")
    rank_0_print(f"  - Language model: {cfg.train_language_model}")
    rank_0_print(f"  - Progress head: {cfg.train_progress_head}")
    rank_0_print(f"  - Success head: {getattr(cfg, 'train_success_head', False)}")
    rank_0_print(f"  - Preference head: {cfg.train_preference_head}")
    rank_0_print(f"  - Similarity head: {cfg.train_similarity_head}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            rank_0_print(f"{name:60} | {param.shape} | RG: {param.requires_grad}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    rank_0_print(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return tokenizer, processor, model


def setup_peft_model(rfm_model: RFM, cfg: PEFTConfig) -> RFM:
    """Shared function to apply PEFT configuration to the model"""

    rank_0_print("Using PEFT/LoRA training...")
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
    )
    if cfg.peft_vision_encoder:
        # vision backbone is frozen, but we can still train the LoRA parameters
        rank_0_print("Attaching LoRA to only the vision encoder...")
        rfm_model.base_model.model.visual = get_peft_model(rfm_model.base_model.model.visual, lora_config)

    # Count trainable parameters manually - defer printing until after FSDP setup
    trainable_params = sum(p.numel() for p in rfm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in rfm_model.parameters())
    rank_0_print(
        f"AFTER PEFT: trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
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
        "weight_decay": cfg.training.weight_decay,
        "disable_tqdm": False,
        # # Compile settings
        # "torch_compile": True,
        # "torch_compile_mode": "max-autotune",
        # "torch_compile_backend": "inductor",
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
            # Disable HuggingFace's automatic logging - we use custom Logger class instead
            "report_to": "none",
        })

    return TrainingArguments(**base_args)


def setup_dataset(cfg: DataConfig, is_eval: bool = False, **kwargs) -> BaseDataset:
    """Shared function to create Dataset for training or evaluation"""
    dataset_cls = {
        "rfm": RFMDataset,
        "data_source_balance": BalancedRFMDataset,
    }
    dataset = dataset_cls[cfg.dataset_type](config=cfg, is_evaluation=is_eval, **kwargs)
    dataset = InfiniteDataset(dataset)
    return dataset


def setup_custom_eval_dataset(cfg: DataConfig, sampler_type: str, is_eval: bool = False, **kwargs):
    """Setup a custom evaluation dataset with a specific sampler."""
    custom_eval_dataset = CustomEvalDataset(sampler_type, cfg, is_evaluation=is_eval, **kwargs)

    return custom_eval_dataset


def setup_batch_collator(
    processor: AutoProcessor, tokenizer: AutoTokenizer, cfg: ExperimentConfig, is_eval: bool = False
) -> BaseCollator:
    """Shared function to create BatchCollator"""
    collator_kwargs = {
        "processor": processor,
        "max_length": cfg.training.max_seq_length,
        "resized_height": cfg.data.resized_height,
        "resized_width": cfg.data.resized_width,
        "base_model_id": cfg.model.base_model_id,
        "use_multi_image": cfg.data.use_multi_image,
        "use_progress_token": cfg.model.use_progress_token,
    }
    if "Qwen" in cfg.model.base_model_id or "SmolVLM" in cfg.model.base_model_id:
        if cfg.model.model_type == "default":
            batch_collator = RFMBatchCollator(**collator_kwargs)
        elif cfg.model.model_type == "vqa":
            batch_collator = VQABatchCollator(**collator_kwargs, inference=is_eval)
    elif "rewind_transformer" in cfg.model.base_model_id:
        batch_collator = ReWiNDBatchCollator(
            **collator_kwargs, tokenizer=tokenizer, load_embeddings=cfg.data.load_embeddings
        )
    return batch_collator
