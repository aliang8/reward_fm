#!/usr/bin/env python3
"""
Script to load an existing model checkpoint and run custom evaluations.

Usage:
    # Using default config
    uv run python run_eval_only.py model_path=rewardfm/rewardfm/ant-rfm-qwen-4gpu-bs12-pref-prog-20251205-132026
    
    # Override config values
    uv run python run_eval_only.py \
        model_path=rewardfm/ant-rfm-qwen-prog-only-images-bs12-prog-only-more-rewind \
        custom_eval.eval_types=[policy_ranking,reward_alignment] \
        custom_eval.reward_alignment=[reward_alignment] \
        custom_eval.policy_ranking=[policy_ranking]

    # Use a different config file
    uv run python run_eval_only.py --config-name my_eval_config model_path=path/to/model
"""

import json
import os
from dataclasses import asdict
from typing import Optional

import torch
import wandb
from hydra import main as hydra_main
from omegaconf import OmegaConf, DictConfig

from rfm.configs.eval_configs import OfflineEvalConfig
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.evals.eval_utils import load_model_from_hf, load_wandb_run_info
from rfm.trainers import RFMHeadsTrainer, RFMVQATrainer, SingleFrameTrainer
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import get_logger
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
from rfm.utils.setup_utils import (
    create_training_arguments,
    setup_batch_collator,
)    

logger = get_logger()

def create_eval_trainer(
    cfg: ExperimentConfig,
    model,
    processor,
    tokenizer,
    output_dir: str,
):
    """Create trainer configured for evaluation only.
    
    Supports RFMHeadsTrainer, SingleFrameTrainer, and RFMVQATrainer based on config.trainer_cls.
    """
    logger.info("Setting up trainer for evaluation...")

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

    # Determine trainer class based on config (check trainer_cls first, then model_type)
    trainer_cls_name = getattr(cfg, "trainer_cls", None) or "rfm_heads"
    
    if trainer_cls_name == "single_frame":
        trainer = SingleFrameTrainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Not needed for eval
            eval_dataset=None,  # Will be created in _run_custom_evaluations
            data_collator=batch_collator,
            config=cfg,
        )
    elif cfg.model.model_type == "vqa":
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


@hydra_main(version_base=None, config_path="rfm/configs", config_name="eval_only_config")
def main(cfg: DictConfig):
    # Convert Hydra config to dataclass
    eval_only_cfg = convert_hydra_to_dataclass(cfg, OfflineEvalConfig)
    
    # Display the evaluation config
    display_config(eval_only_cfg)

    # Validate model_path is provided
    if not eval_only_cfg.model_path:
        raise ValueError("model_path is required. Provide a HuggingFace model ID or local checkpoint path.")

    model_path = eval_only_cfg.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment config and model using load_model_from_hf
    # This function handles both HuggingFace repos and local paths
    logger.info(f"Loading model and config from: {model_path}")
    exp_cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device)
    logger.info(f"✅ Model and config loaded successfully from {model_path}")

    # Display the experiment config
    display_config(exp_cfg)

    # Try to load existing wandb info from model path
    wandb_info = load_wandb_run_info(model_path)
    resume_id = None
    if wandb_info:
        resume_id = wandb_info.get("wandb_id")
        if resume_id:
            logger.info(f"Found existing wandb run ID: {resume_id}, will resume run")

    
    # Initialize wandb if enabled in experiment config (only on rank 0)
    if "wandb" in exp_cfg.logging.log_to and is_rank_0():
        config_dict = asdict(exp_cfg)
        model_name = model_path.replace("/", "_") if "/" in model_path else model_path
        init_kwargs = {
            "project": exp_cfg.logging.wandb_project,
            "entity": exp_cfg.logging.wandb_entity,
            "name": f"eval_{model_name}",
            "config": config_dict,
        }
        if exp_cfg.logging.wandb_notes:
            init_kwargs["notes"] = exp_cfg.logging.wandb_notes
        # Resume existing run if resume_id is found
        if resume_id:
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = "must"
        wandb.init(**init_kwargs)
        if resume_id:
            logger.info(f"Wandb resumed run: eval_{model_name} (ID: {resume_id})")
        else:
            logger.info(f"Wandb initialized for evaluation: eval_{model_name}")
        if exp_cfg.logging.wandb_notes:
            logger.info(f"Wandb notes: {exp_cfg.logging.wandb_notes}")
    elif "wandb" in exp_cfg.logging.log_to:
        logger.info("Wandb logging enabled but skipped on non-rank-0 processes")

    # Determine output directory
    output_dir = eval_only_cfg.output_dir
    if output_dir is None:
        # Use model name as output dir (sanitize path for both HF repos and local paths)
        model_name = model_path.replace("/", "_").replace("@", "_")
        output_dir = os.path.join("./eval_output", model_name)

    # Override custom_eval settings from OfflineEvalConfig
    exp_cfg.custom_eval = eval_only_cfg.custom_eval

    # Create trainer
    trainer = create_eval_trainer(exp_cfg, model, processor, tokenizer, output_dir)

    # Set output_dir in config for video saving
    exp_cfg.output_dir = output_dir

    # Ensure model is in eval mode
    trainer.model.eval()

    # Run custom evaluations
    # This method creates datasets internally based on config.custom_eval settings
    metrics = trainer._run_custom_evaluations(output_dir=output_dir)

    # Save evaluation metrics to JSON file
    if metrics and is_rank_0():
        metrics_file = os.path.join(output_dir, "eval_metrics.json")
        # Convert any numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_serializable[k] = float(v)
            elif isinstance(v, (list, dict)):
                metrics_serializable[k] = v
            else:
                # Try to convert to float if possible
                try:
                    metrics_serializable[k] = float(v)
                except (ValueError, TypeError):
                    metrics_serializable[k] = str(v)
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"💾 Saved evaluation metrics to: {metrics_file}")

    logger.info("\n✅ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
