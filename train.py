import json
import os
import warnings
from dataclasses import asdict
import shutil

import torch
import torch.distributed as dist
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from peft import prepare_model_for_kbit_training
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.trainers import ReWiNDTrainer, RFMHeadsTrainer, RFMVQATrainer
from rfm.data.datasets.helpers import show_available_datasets
from rfm.utils.distributed import is_rank_0, rank_0_print
from rfm.utils.timer import _timer
from rfm.utils.parser import parse_multiple
from rfm.utils.save import SaveBestCallback
from rfm.utils.setup_utils import (
    create_training_arguments,
    setup_batch_collator,
    setup_dataset,
    setup_model_and_processor,
    setup_peft_model,
)
from rfm.utils.logger import Logger
import datasets

datasets.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)


def train(cfg: ExperimentConfig):
    timing_raw = {}

    run_name = cfg.training.exp_name
    if cfg.debug:
        run_name += "_debug"

    # will initialize wandb later via logger (after logger is constructed)

    # Set memory management
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use the shared function to set up model and processor
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw):
        tokenizer, processor, rfm_model = setup_model_and_processor(cfg.model, peft_config=cfg.peft)

    # Apply PEFT if enabled
    if cfg.model.use_peft:
        peft_rfm_model = setup_peft_model(rfm_model, cfg.peft)
    else:
        peft_rfm_model = rfm_model
        rank_0_print("PEFT not enabled, using full model")

    if cfg.model.quantization:
        peft_rfm_model = prepare_model_for_kbit_training(peft_rfm_model)

    # Create training arguments from config
    if cfg.debug:
        cfg.training.logging_steps = 2
        cfg.training.eval_steps = 2
        cfg.data.eval_subset_size = 10
        cfg.training.custom_eval_steps = 2

    output_dir = os.path.join(cfg.training.output_dir, run_name)

    training_args = create_training_arguments(cfg, output_dir)

    # Handle output directory existence (works with accelerate/distributed training)
    overwrite_output_dir = getattr(cfg.training, "overwrite_output_dir", False)

    # Check if distributed training is initialized (for proper synchronization)
    # This is important for accelerate/FSDP setups where multiple processes run
    dist_initialized = dist.is_available() and dist.is_initialized()

    # Check if output directory exists (only on rank 0 to avoid race conditions)
    if is_rank_0() and os.path.exists(output_dir):
        if overwrite_output_dir:
            rank_0_print(f"Output directory {output_dir} already exists. Overwriting (overwrite_output_dir=True)...")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                f"Output directory {output_dir} already exists. "
                f"Set overwrite_output_dir=True in config to overwrite it, or use a different output directory."
            )

    # Synchronize all processes before creating directory (important for distributed training)
    # This ensures rank 0 finishes checking/removing before other processes try to create it
    if dist_initialized:
        dist.barrier()

    # Create output directory (all processes need to do this for distributed training)
    # os.makedirs is safe to call multiple times (exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Synchronize after directory creation to ensure all processes see it
    if dist_initialized:
        dist.barrier()

    # Initialize logger (works with wandb/tensorboard)
    log_to = cfg.logging.log_to
    logger = Logger(log_to=log_to, output_dir=output_dir, is_main_process=is_rank_0())
    config_save_path = os.path.join(output_dir, "config.yaml")
    config_dict = asdict(cfg)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    rank_0_print(f"Saved training config to: {config_save_path}")

    # Initialize wandb via logger if requested
    if "wandb" in (cfg.logging.log_to or []) and is_rank_0():
        # Convert config to dict for wandb using dataclass asdict
        config_dict = asdict(cfg)
        logger.init_wandb(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
            config=config_dict,
        )
        rank_0_print(f"Wandb initialized: {run_name}")

    logger.write_wandb_info(output_dir, run_name)

    # Use the shared utilities for batch collator and dataset

    if is_rank_0():
        show_available_datasets()

    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=False)
        train_dataset = setup_dataset(cfg.data, batch_size=cfg.training.per_device_train_batch_size)

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
        "rewind_scale_transformer": ReWiNDTrainer,
    }[cfg.trainer_cls]
    rank_0_print(f"Trainer class: {trainer_cls}")

    # Add SaveBestCallback to automatically save and upload best models
    save_best_cfg = cfg.logging.save_best
    save_callback = SaveBestCallback(
        metric_names=save_best_cfg.metric_names,
        greater_is_better=save_best_cfg.greater_is_better,
        keep_top_k=save_best_cfg.keep_top_k,
        upload_to_hub=save_best_cfg.upload_to_hub,
        hub_token=save_best_cfg.hub_token,
        hub_private=save_best_cfg.hub_private,
        base_model=cfg.model.base_model_id,
    )

    trainer = trainer_cls(
        model=peft_rfm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=batch_collator,
        config=cfg,
        logger=logger,
        callbacks=[save_callback],
    )

    # Set trainer reference in the callback so it can access trainer methods
    save_callback.setup_trainer_reference(trainer)

    # Debug: Check if callback was added
    rank_0_print(f"🔧 DEBUG: Trainer callbacks: {[type(cb).__name__ for cb in trainer.callback_handler.callbacks]}")

    metrics_info = []
    for name, is_better in zip(save_best_cfg.metric_names, save_best_cfg.greater_is_better):
        direction = "↗️ higher" if is_better else "↘️ lower"
        metrics_info.append(f"{name} ({direction})")

    rank_0_print(f"💾 SaveBest monitoring: {', '.join(metrics_info)}")
    rank_0_print(f"📁 Keeping top {save_best_cfg.keep_top_k} checkpoint(s) and upload(s)")

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

    # log timing_raw via logger
    if is_rank_0():
        logger.log_scalars(timing_raw)

    rank_0_print(f"Timing raw: {timing_raw}")
    rank_0_print(f"Training from checkpoint: {cfg.training.resume_from_checkpoint}")

    if cfg.debug:
        rank_0_print("🐛 DEBUG MODE: eval_steps=2, custom_eval_steps=2, eval_subset_size=10")

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
