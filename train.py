import json
import os
import warnings
from dataclasses import asdict

import torch
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

import wandb

# Import shared configs and utilities
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.trainers import ReWiNDTrainer, RFMHeadsTrainer, RFMVQATrainer
from rfm.utils.distributed import is_rank_0, rank_0_print
from rfm.utils.timer import _timer
from rfm.utils.parser import parse_multiple
from rfm.utils.setup_utils import (
    create_training_arguments,
    setup_batch_collator,
    setup_dataset,
    setup_model_and_processor,
    setup_peft_model,
)

# Suppress FSDP ShardedTensor deprecation warning
warnings.filterwarnings("ignore", message="Please use DTensor instead and we are deprecating ShardedTensor")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(cfg: ExperimentConfig):
    timing_raw = {}

    run_name = f"{cfg.logging.wandb_run_name}"
    if cfg.debug:
        run_name += "_debug"

    # Initialize wandb if enabled (only on rank 0)
    if cfg.logging.use_wandb and is_rank_0():
        # Convert config to dict for wandb using dataclass asdict
        config_dict = asdict(cfg)

        wandb.init(
            project=cfg.logging.wandb_project, entity=cfg.logging.wandb_entity, name=run_name, config=config_dict
        )
        rank_0_print(f"Wandb initialized: {wandb.run.name}")
    elif cfg.logging.use_wandb:
        rank_0_print("Wandb logging enabled but skipped on non-rank-0 processes")

    # Set memory management
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use the shared function to set up model and processor
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw):
        tokenizer, processor, rfm_model = setup_model_and_processor(cfg.model)

    # Apply PEFT if enabled
    if cfg.model.use_peft:
        peft_rfm_model = setup_peft_model(rfm_model, cfg.peft)
    else:
        peft_rfm_model = rfm_model
        rank_0_print("PEFT not enabled, using full model")

    # Create training arguments from config
    if cfg.debug:
        cfg.training.save_steps = 2
        cfg.training.logging_steps = 2
        cfg.training.eval_steps = 2
        cfg.data.eval_subset_size = 10

    training_args = create_training_arguments(cfg, cfg.training.output_dir)

    # Save config to output directory
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.training.output_dir, "config.yaml")
    config_dict = asdict(cfg)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    rank_0_print(f"Saved training config to: {config_save_path}")

    # Save wandb run ID to a file for later reference (only on rank 0)
    if cfg.logging.use_wandb and is_rank_0() and wandb.run is not None:
        wandb_info = {
            "wandb_id": wandb.run.id,
            "wandb_name": wandb.run.name,
            "wandb_project": wandb.run.project,
            "wandb_entity": wandb.run.entity,
            "wandb_url": wandb.run.url,
        }
        wandb_info_path = os.path.join(cfg.training.output_dir, "wandb_info.json")
        with open(wandb_info_path, "w") as f:
            json.dump(wandb_info, f, indent=2)
        rank_0_print(f"Saved wandb run info to: {wandb_info_path}")
        rank_0_print(f"Wandb ID: {wandb.run.id}")

    # Use the shared utilities for batch collator and dataset
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_batch_collator(processor, tokenizer, cfg)
        train_dataset = setup_dataset(cfg.data)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        dataset_kwargs = {"max_samples": cfg.data.eval_subset_size}

        eval_dataset = setup_dataset(cfg.data, is_eval=True, **dataset_kwargs)
        rank_0_print(f"Evaluation dataset created with {cfg.data.eval_subset_size} samples")

    trainer_cls = {
        "rfm_heads": RFMHeadsTrainer,
        "rewind_transformer": ReWiNDTrainer,
        "rfm_vqa": RFMVQATrainer,
    }[cfg.trainer_cls]
    rank_0_print(f"Trainer class: {trainer_cls}")

    trainer = trainer_cls(
        model=peft_rfm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=batch_collator,
        config=cfg,
    )
    if is_rank_0():
        print("\n" + "=" * 80)
        print("--- PRE-TRAINING FSDP DIAGNOSTICS ---")
        # The Trainer creates its own Accelerator instance. Let's check its state.
        if hasattr(trainer, "accelerator"):
            print("Trainer's Accelerator object found.")
            fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin:
                print("FSDP Plugin found in Accelerator state.")
                # This is the configuration the accelerator will ACTUALLY use for wrapping.
                print(f"VERIFY: Actual FSDP plugin config being used: {fsdp_plugin}")
            else:
                print("ERROR: FSDP Plugin NOT found in the Trainer's accelerator state!")
        else:
            print("ERROR: Trainer has no 'accelerator' attribute yet. This check needs to be later.")
        print("=" * 80 + "\n")

    # log timing_raw to wandb
    if cfg.logging.use_wandb and is_rank_0():
        wandb.log(timing_raw)

    rank_0_print(f"Timing raw: {timing_raw}")
    rank_0_print(f"Training from checkpoint: {cfg.training.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(cfg.training.output_dir)
    rank_0_print(f"Training complete! Check {cfg.training.output_dir} for checkpoints and final model.")


def display_config(cfg: ExperimentConfig):
    """Display the configuration in a nice Rich format."""
    if not is_rank_0():
        return  # Only display config on rank 0

    console = Console()
    console.print(cfg)


def main(cfg: ExperimentConfig):
    # Display the configuration in a nice Rich format
    display_config(cfg)

    if cfg.mode == "train":
        if is_rank_0():
            rprint(Panel.fit("🚀 Starting RFM Training", style="bold green"))
        train(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'train' or 'evaluate'")


if __name__ == "__main__":
    cfg = parse_multiple(ExperimentConfig)
    main(cfg)
