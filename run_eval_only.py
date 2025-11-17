#!/usr/bin/env python3
"""
Script to load an existing model checkpoint and run custom evaluations.

Usage:
    # Using command line arguments with HuggingFace model ID
    uv run python run_eval_only.py \
        --config_path rfm/configs/eval_only_config.yaml \
        --model_path rewardfm/ant-rfm-qwen-prog-only-images-bs12-prog-only-more-rewind
"""

import os
import yaml
import torch
from pathlib import Path
from dataclasses import asdict
import argparse
import ast

from transformers import TrainingArguments
from rfm.configs.experiment_configs import ExperimentConfig, CustomEvaluationConfig
from rfm.configs.eval_configs import EvalOnlyConfig
from rfm.trainers import RFMHeadsTrainer, RFMVQATrainer
from rfm.utils.setup_utils import (
    setup_model_and_processor,
    setup_batch_collator,
    create_training_arguments,
)
from rfm.utils.distributed import is_rank_0, rank_0_print
from rfm.utils.parser import parse_multiple
import wandb


def load_config_from_yaml(config_path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    rank_0_print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        yaml_text = f.read()

    # Some configs (e.g., ReWiND-style) serialize custom python objects in YAML.
    # Provide a custom loader that safely converts those tags into plain dicts.
    class _EvalOnlyConfigLoader(yaml.SafeLoader):
        pass

    _EvalOnlyConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/object:rfm.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    try:
        config_dict = yaml.load(yaml_text, Loader=_EvalOnlyConfigLoader)
    except Exception as e:
        rank_0_print(f"Warning: Custom YAML loading failed ({e}), falling back to safe_load")
        config_dict = yaml.safe_load(yaml_text)
    
    # Create config from dictionary
    cfg = ExperimentConfig(**config_dict)
    return cfg


def load_model_from_checkpoint(cfg: ExperimentConfig, model_path: str):
    """Load model and processor from HuggingFace model ID or local checkpoint path."""
    rank_0_print(f"Loading model from: {model_path}")
    
    # model_path can be either a HuggingFace model ID or a local path
    # setup_model_and_processor handles both cases
    hf_model_id = model_path
    
    # Load model and processor
    # Pass the model_path to setup_model_and_processor which handles loading
    # If PEFT was used during training, the checkpoint should contain adapter weights
    # which will be automatically loaded by from_pretrained
    tokenizer, processor, rfm_model = setup_model_and_processor(
        cfg.model, 
        hf_model_id=hf_model_id, 
        peft_config=cfg.peft if cfg.model.use_peft else None
    )
    
    # Note: If PEFT was used during training, the adapter weights should already be loaded
    # by from_pretrained. We don't need to call setup_peft_model again for evaluation.
    
    rank_0_print(f"✅ Model loaded successfully from {model_path}")
    return tokenizer, processor, rfm_model


def create_eval_trainer(
    cfg: ExperimentConfig,
    model,
    processor,
    tokenizer,
    output_dir: str,
):
    """Create RFMHeadsTrainer configured for evaluation only."""
    rank_0_print("Setting up trainer for evaluation...")
    
    # Create minimal training arguments (needed for trainer initialization)
    # Most settings don't matter since we're only evaluating
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training args with eval settings
    training_args = create_training_arguments(cfg, output_dir)
    
    # Override some settings for evaluation-only
    training_args.do_train = False
    training_args.do_eval = False  # Disable default evaluation since we'll use custom evaluations
    training_args.eval_strategy = "no"  # We'll call _run_custom_evaluations() manually
    # Also set evaluation_strategy for backwards compatibility
    if hasattr(training_args, 'evaluation_strategy'):
        training_args.evaluation_strategy = "no"
    
    # Set up batch collator
    batch_collator = setup_batch_collator(processor, tokenizer, cfg)

    if cfg.model.model_type == "vqa":
        trainer = RFMVQATrainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Not needed for eval
            eval_dataset=None,   # Will be created in _run_custom_evaluations
            data_collator=batch_collator,
            config=cfg,
        )
    else:
        trainer = RFMHeadsTrainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Not needed for eval
            eval_dataset=None,   # Will be created in _run_custom_evaluations
            data_collator=batch_collator,
            config=cfg,
        )
    
    return trainer


def run_custom_evaluations(trainer: RFMHeadsTrainer):
    """Run custom evaluations using the trainer."""
    rank_0_print("=" * 100)
    rank_0_print("Starting custom evaluations...")
    rank_0_print("=" * 100)
    
    # Ensure model is in eval mode
    trainer.model.eval()
    
    # Run custom evaluations
    # This method creates datasets internally based on config.custom_eval settings
    custom_metrics = trainer._run_custom_evaluations()
    
    rank_0_print("=" * 100)
    rank_0_print("Custom evaluations completed!")
    rank_0_print("=" * 100)
    
    # Print metrics summary
    if is_rank_0():
        rank_0_print("\nEvaluation Metrics Summary:")
        for metric_name, metric_value in custom_metrics.items():
            rank_0_print(f"  {metric_name}: {metric_value}")
    
    return custom_metrics


def main():
    # Parse EvalOnlyConfig using pyrallis (supports --config_paths and CLI overrides)
    eval_only_cfg = parse_multiple(EvalOnlyConfig)
    
    # Validate model_path is provided
    if not eval_only_cfg.model_path:
        raise ValueError("model_path is required. Provide a HuggingFace model ID or local checkpoint path.")
    
    model_path = eval_only_cfg.model_path
    
    # Initialize wandb if enabled (only on rank 0)
    # This needs to happen early so that trainer's wandb logging works
    if is_rank_0():
        # Check if wandb logging is enabled in the experiment config (will be loaded later)
        # For now, we'll initialize wandb if it's requested, but we'll get the actual config after loading
        # We can initialize wandb later after loading exp_cfg, or just disable wandb for eval-only runs
        # Let's disable wandb for standalone eval runs unless explicitly configured
        pass  # We'll handle wandb initialization after loading exp_cfg
    
    # Determine if model_path is a HuggingFace ID or local path
    is_hf_repo = "/" in model_path and not os.path.exists(model_path) and not model_path.startswith("/")
    is_local_path = os.path.exists(model_path) or os.path.isdir(model_path)
    
    # Load experiment config (training config)
    if eval_only_cfg.exp_config_path:
        # Load from provided config file
        exp_cfg = load_config_from_yaml(eval_only_cfg.exp_config_path)
        rank_0_print(f"Loaded experiment config from: {eval_only_cfg.exp_config_path}")
    elif is_local_path:
        # Try to load from local checkpoint directory
        config_path = os.path.join(model_path, "config.yaml")
        outside_config_path = os.path.join(model_path, "..", "config.yaml")
        if os.path.exists(config_path):
            exp_cfg = load_config_from_yaml(config_path)
            rank_0_print(f"Loaded experiment config from local checkpoint: {config_path}")
        elif os.path.exists(outside_config_path):
            exp_cfg = load_config_from_yaml(outside_config_path)
            rank_0_print(f"Loaded experiment config from local checkpoint: {outside_config_path}")
        else:
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Please provide exp_config_path or ensure checkpoint contains config.yaml"
            )
    elif is_hf_repo:
        # Try to load config.yaml from HuggingFace repo
        from huggingface_hub import hf_hub_download
        try:
            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.yaml",
                local_files_only=False,
            )
            exp_cfg = load_config_from_yaml(config_path)
            rank_0_print(f"Loaded experiment config from HuggingFace repo: {model_path}/config.yaml")
        except Exception as e:
            rank_0_print(f"Could not load config.yaml from HuggingFace repo {model_path}: {e}")
            raise FileNotFoundError(
                f"Could not load config.yaml from HuggingFace repo {model_path}. "
                f"Please provide exp_config_path or ensure the repo contains config.yaml"
            )
    else:
        raise ValueError(
            f"Could not determine if {model_path} is a HuggingFace repo or local path. "
            f"Please provide exp_config_path or ensure the path is valid."
        )
    
    # Initialize wandb if enabled in experiment config (only on rank 0)
    if "wandb" in exp_cfg.logging.log_to and is_rank_0():
        from dataclasses import asdict
        config_dict = asdict(exp_cfg)
        model_name = model_path.replace("/", "_") if "/" in model_path else model_path
        wandb.init(
            project=exp_cfg.logging.wandb_project,
            entity=exp_cfg.logging.wandb_entity,
            name=f"eval_{model_name}",
            config=config_dict
        )
        rank_0_print(f"Wandb initialized for evaluation: eval_{model_name}")
    elif "wandb" in exp_cfg.logging.log_to:
        rank_0_print("Wandb logging enabled but skipped on non-rank-0 processes")
    
    # Merge custom_eval from EvalOnlyConfig if provided
    # Only override fields that are explicitly set (non-empty lists)
    if eval_only_cfg.custom_eval.eval_types:
        exp_cfg.custom_eval.eval_types = eval_only_cfg.custom_eval.eval_types
        rank_0_print(f"Using eval_types from EvalOnlyConfig: {eval_only_cfg.custom_eval.eval_types}")
    
    # Override specific eval dataset lists if provided
    for eval_type in ['reward_alignment', 'policy_ranking', 'confusion_matrix']:
        eval_datasets = getattr(eval_only_cfg.custom_eval, eval_type, None)
        if eval_datasets and len(eval_datasets) > 0:
            setattr(exp_cfg.custom_eval, eval_type, eval_datasets)
            rank_0_print(f"Using {eval_type} datasets from EvalOnlyConfig: {eval_datasets}")
    
    # Verify model path exists (if local) or is accessible (if HuggingFace)
    if is_local_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Ensure custom eval is configured
    if not hasattr(exp_cfg.custom_eval, 'eval_types') or not exp_cfg.custom_eval.eval_types:
        rank_0_print("Warning: No eval_types configured in custom_eval. Please set custom_eval.eval_types in your config.")
        return
    
    # Determine output directory
    output_dir = eval_only_cfg.output_dir
    if output_dir is None:
        # Use model name as output dir for HuggingFace repos, or checkpoint path for local
        if is_hf_repo:
            model_name = model_path.replace("/", "_")
            output_dir = os.path.join("./eval_output", model_name)
        else:
            output_dir = os.path.join(model_path, "eval_output")
    
    # Load model
    tokenizer, processor, model = load_model_from_checkpoint(exp_cfg, model_path)
    
    # Create trainer
    trainer = create_eval_trainer(exp_cfg, model, processor, tokenizer, output_dir)
    
    # Run custom evaluations
    metrics = run_custom_evaluations(trainer)
    
    rank_0_print("\n✅ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()

