#!/usr/bin/env python3
"""
Script to load an existing model checkpoint and run custom evaluations.

Usage:
    # Using default config
    uv run python run_eval_only.py model_path=rewardfm/rewardfm/ant-rfm-qwen-4gpu-bs12-pref-prog-20251205-132026
    
    # Override config values
    uv run python run_eval_only.py \
        model_path=rewardfm/ant-rfm-qwen-prog-only-images-bs12-prog-only-more-rewind \
        custom_eval.eval_types=[policy_ranking,reward_alignment]
    
    # Use a different config file
    uv run python run_eval_only.py --config-name my_eval_config model_path=path/to/model
"""

import os
import torch
from typing import Optional

from omegaconf import OmegaConf, DictConfig
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore

from rfm.configs.experiment_configs import ExperimentConfig
from rfm.configs.eval_configs import OfflineEvalConfig
from rfm.trainers import RFMHeadsTrainer, RFMVQATrainer
from rfm.utils.setup_utils import (
    setup_batch_collator,
    create_training_arguments,
)
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import rank_0_info, get_logger
import wandb
from rfm.evals.eval_utils import load_model_from_hf

logger = get_logger()

# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="eval_only_config", node=OfflineEvalConfig)


def create_eval_trainer(
    cfg: ExperimentConfig,
    model,
    processor,
    tokenizer,
    output_dir: str,
):
    """Create RFMHeadsTrainer configured for evaluation only."""
    rank_0_info("Setting up trainer for evaluation...")

    # Create minimal training arguments (needed for trainer initialization)
    # Most settings don't matter since we're only evaluating
    os.makedirs(output_dir, exist_ok=True)

    # Create training args with eval settings
    training_args = create_training_arguments(cfg.training, output_dir)

    # Override some settings for evaluation-only
    training_args.do_train = False
    training_args.do_eval = False  # Disable default evaluation since we'll use custom evaluations
    training_args.eval_strategy = "no"  # We'll call _run_custom_evaluations() manually
    # Also set evaluation_strategy for backwards compatibility
    if hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "no"

    # Set up batch collator
    batch_collator = setup_batch_collator(processor, tokenizer, cfg)

    if cfg.model.model_type == "vqa":
        trainer = RFMVQATrainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Not needed for eval
            eval_dataset=None,  # Will be created in _run_custom_evaluations
            data_collator=batch_collator,
            config=cfg,
        )
    else:
        trainer = RFMHeadsTrainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Not needed for eval
            eval_dataset=None,  # Will be created in _run_custom_evaluations
            data_collator=batch_collator,
            config=cfg,
        )

    return trainer


def run_custom_evaluations(trainer: RFMHeadsTrainer):
    """Run custom evaluations using the trainer."""
    rank_0_info("=" * 100)
    rank_0_info("Starting custom evaluations...")
    rank_0_info("=" * 100)

    # Ensure model is in eval mode
    trainer.model.eval()

    # Run custom evaluations
    # This method creates datasets internally based on config.custom_eval settings
    custom_metrics = trainer._run_custom_evaluations()

    rank_0_info("=" * 100)
    rank_0_info("Custom evaluations completed!")
    rank_0_info("=" * 100)

    # Print metrics summary
    if is_rank_0():
        rank_0_info("\nEvaluation Metrics Summary:")
        for metric_name, metric_value in custom_metrics.items():
            rank_0_info(f"  {metric_name}: {metric_value}")

    return custom_metrics


def convert_hydra_to_dataclass(cfg: DictConfig) -> OfflineEvalConfig:
    """Convert Hydra DictConfig to OfflineEvalConfig dataclass."""
    if OmegaConf.is_struct(cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode="convert")
    else:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return OfflineEvalConfig(**cfg_dict)


@hydra_main(version_base=None, config_path="rfm/configs", config_name="eval_only_config")
def main(cfg: DictConfig):
    # Convert Hydra config to dataclass
    eval_only_cfg = convert_hydra_to_dataclass(cfg)

    # Validate model_path is provided
    if not eval_only_cfg.model_path:
        raise ValueError("model_path is required. Provide a HuggingFace model ID or local checkpoint path.")

    model_path = eval_only_cfg.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment config and model using load_model_from_hf
    # This function handles both HuggingFace repos and local paths
    rank_0_info(f"Loading model and config from: {model_path}")
    exp_cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device)
    rank_0_info(f"✅ Model and config loaded successfully from {model_path}")

    # Initialize wandb if enabled in experiment config (only on rank 0)
    if "wandb" in exp_cfg.logging.log_to and is_rank_0():
        from dataclasses import asdict

        config_dict = asdict(exp_cfg)
        model_name = model_path.replace("/", "_") if "/" in model_path else model_path
        wandb.init(
            project=exp_cfg.logging.wandb_project,
            entity=exp_cfg.logging.wandb_entity,
            name=f"eval_{model_name}",
            config=config_dict,
        )
        rank_0_info(f"Wandb initialized for evaluation: eval_{model_name}")
    elif "wandb" in exp_cfg.logging.log_to:
        rank_0_info("Wandb logging enabled but skipped on non-rank-0 processes")

    # Merge custom_eval from OfflineEvalConfig if provided
    # Only override fields that are explicitly set (non-empty lists)
    if eval_only_cfg.custom_eval.eval_types:
        exp_cfg.custom_eval.eval_types = eval_only_cfg.custom_eval.eval_types
        rank_0_info(f"Using eval_types from OfflineEvalConfig: {eval_only_cfg.custom_eval.eval_types}")

    # Override specific eval dataset lists if provided
    for eval_type in ["reward_alignment", "policy_ranking", "confusion_matrix"]:
        eval_datasets = getattr(eval_only_cfg.custom_eval, eval_type, None)
        if eval_datasets and len(eval_datasets) > 0:
            setattr(exp_cfg.custom_eval, eval_type, eval_datasets)
            rank_0_info(f"Using {eval_type} datasets from OfflineEvalConfig: {eval_datasets}")

    # Ensure custom eval is configured
    if not hasattr(exp_cfg.custom_eval, "eval_types") or not exp_cfg.custom_eval.eval_types:
        logger.warning(
            "No eval_types configured in custom_eval. Please set custom_eval.eval_types in your config."
        )
        return

    # Determine output directory
    output_dir = eval_only_cfg.output_dir
    if output_dir is None:
        # Use model name as output dir (sanitize path for both HF repos and local paths)
        model_name = model_path.replace("/", "_").replace("@", "_")
        output_dir = os.path.join("./eval_output", model_name)

    # Create trainer
    trainer = create_eval_trainer(exp_cfg, model, processor, tokenizer, output_dir)

    # Run custom evaluations
    metrics = run_custom_evaluations(trainer)

    rank_0_info("\n✅ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
