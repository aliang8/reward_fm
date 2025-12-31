import json
import os
from dataclasses import asdict
import shutil

import torch
import torch.distributed as dist
import yaml
from rich import print as rprint
from rich.panel import Panel
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra import main as hydra_main

from peft import prepare_model_for_kbit_training
from rfm.configs.experiment_configs import (
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    DataConfig,
    TrainingConfig,
    LossConfig,
    LoggingConfig,
    SaveBestConfig,
    CustomEvaluationConfig,
)
from rfm.trainers import ReWiNDTrainer, RFMHeadsTrainer, RFMVQATrainer
from rfm.data.datasets.helpers import show_available_datasets
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import rank_0_info
from rfm.utils.timer import _timer
from rfm.utils.save import SaveBestCallback, resolve_checkpoint_path
from rfm.utils.setup_utils import (
    create_training_arguments,
    setup_batch_collator,
    setup_dataset,
    setup_model_and_processor,
    setup_peft_model,
)
from rfm.data.datasets.base import resolve_dataset_keys
from rfm.utils.logger import Logger
from rfm.utils.distributed import banner
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
import datasets

datasets.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
cs.store(group="model", name="model_config", node=ModelConfig)
cs.store(group="peft", name="peft_config", node=PEFTConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="training", name="training_config", node=TrainingConfig)
cs.store(group="loss", name="loss_config", node=LossConfig)
cs.store(group="logging", name="logging_config", node=LoggingConfig)
cs.store(group="logging/save_best", name="save_best_config", node=SaveBestConfig)
cs.store(group="custom_eval", name="custom_eval_config", node=CustomEvaluationConfig)


import torch

torch.set_num_threads(64)
torch.set_num_interop_threads(8)


def train(cfg: ExperimentConfig):
    timing_raw = {}

    run_name = cfg.training.exp_name
    if cfg.debug:
        run_name += "_debug"
        cfg.training.logging_steps = 5
        cfg.training.eval_steps = 5
        # cfg.data.eval_subset_size = 100
        cfg.training.custom_eval_steps = 5
        cfg.logging.save_best.save_every = 5
        cfg.data.dataloader_num_workers = 0
        cfg.data.dataloader_persistent_workers = False

        cfg.custom_eval.num_examples_per_quality_pr = 1
        cfg.custom_eval.policy_ranking_max_tasks = 10

    # Set memory management
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    banner("Setting up model and processor")
    # Use the shared function to set up model and processor
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw):
        tokenizer, processor, rfm_model = setup_model_and_processor(cfg.model, peft_config=cfg.peft)

    # Apply PEFT if enabled
    if cfg.model.use_peft:
        peft_rfm_model = setup_peft_model(rfm_model, cfg.peft)
    else:
        peft_rfm_model = rfm_model
        rank_0_info("PEFT not enabled, using full model")

    if cfg.model.quantization:
        peft_rfm_model = prepare_model_for_kbit_training(peft_rfm_model)

    output_dir = os.path.join(cfg.training.output_dir, run_name)

    training_args = create_training_arguments(cfg.training, output_dir)

    # Handle output directory existence (works with accelerate/distributed training)
    overwrite_output_dir = getattr(cfg.training, "overwrite_output_dir", False)

    # Check if distributed training is initialized (for proper synchronization)
    # This is important for accelerate/FSDP setups where multiple processes run
    dist_initialized = dist.is_available() and dist.is_initialized()

    # Check if output directory exists (only on rank 0 to avoid race conditions)
    if is_rank_0() and os.path.exists(output_dir):
        if overwrite_output_dir:
            rank_0_info(f"Output directory {output_dir} already exists. Overwriting (overwrite_output_dir=True)...")
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

    banner("Creating output directory", f"Logging to: {output_dir}")
    # Create output directory (all processes need to do this for distributed training)
    # os.makedirs is safe to call multiple times (exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Synchronize after directory creation to ensure all processes see it
    if dist_initialized:
        dist.barrier()

    # Initialize logger (works with wandb/tensorboard)
    log_to = cfg.logging.log_to
    log_level = cfg.logging.log_level
    logger = Logger(log_to=log_to, output_dir=output_dir, is_main_process=is_rank_0(), log_level=log_level)
    config_save_path = os.path.join(output_dir, "config.yaml")
    config_dict = asdict(cfg)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    rank_0_info(f"Saved training config to: {config_save_path}")

    # Try to load existing wandb info if resuming training
    wandb_info_path = os.path.join(output_dir, "wandb_info.json")
    resume_id = None
    if os.path.exists(wandb_info_path):
        try:
            with open(wandb_info_path) as f:
                wandb_info = json.load(f)
            resume_id = wandb_info.get("wandb_id")
            if resume_id:
                rank_0_info(f"Found existing wandb run ID: {resume_id}, will resume run")
        except Exception as e:
            rank_0_info(f"Could not load wandb info: {e}")

    # Initialize wandb via logger if requested
    if "wandb" in (cfg.logging.log_to or []) and is_rank_0():
        # Convert config to dict for wandb using dataclass asdict
        config_dict = asdict(cfg)
        logger.init_wandb(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
            config=config_dict,
            notes=cfg.logging.wandb_notes,
            mode=cfg.logging.wandb_mode,
            resume_id=resume_id,
        )
        if resume_id:
            rank_0_info(f"Wandb resumed run: {run_name} (ID: {resume_id})")
        else:
            rank_0_info(f"Wandb initialized: {run_name}")
        if cfg.logging.wandb_notes:
            rank_0_info(f"Wandb notes: {cfg.logging.wandb_notes}")

    logger.write_wandb_info(output_dir, run_name)

    # Save wandb run ID to a file for later reference (only on rank 0)
    if cfg.logging.use_wandb and is_rank_0() and wandb.run is not None:
        wandb_info = {
            "wandb_id": wandb.run.id,
            "wandb_name": run_name,
            "wandb_project": wandb.run.project,
            "wandb_entity": wandb.run.entity,
            "wandb_url": wandb.run.url,
        }
        wandb_info_path = os.path.join(output_dir, "wandb_info.json")
        with open(wandb_info_path, "w") as f:
            json.dump(wandb_info, f, indent=2)
        rank_0_print(f"Saved wandb run info to: {wandb_info_path}")
        rank_0_print(f"Wandb ID: {wandb.run.id}")

    # Use the shared utilities for batch collator and dataset

    if is_rank_0():
        show_available_datasets()

    banner("Resolving dataset keys")
    cfg.data.train_datasets = resolve_dataset_keys(cfg.data.train_datasets, split="train")
    rank_0_info(f"Resolved train datasets: {cfg.data.train_datasets}")

    if cfg.data.eval_datasets:
        cfg.data.eval_datasets = resolve_dataset_keys(cfg.data.eval_datasets, split="eval")
        rank_0_info(f"Resolved eval datasets: {cfg.data.eval_datasets}")

    # Resolve custom evaluation dataset keys once (replace in place)
    for eval_type in cfg.custom_eval.eval_types:
        datasets = getattr(cfg.custom_eval, eval_type, None)
        if datasets:
            resolved = resolve_dataset_keys(datasets, split="eval")
            setattr(cfg.custom_eval, eval_type, resolved)
            rank_0_info(f"Resolved {eval_type} datasets: {resolved}")

    rank_0_info("Dataset keys resolved")

    banner("Setting up training and evaluation datasets and collator")
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=False)
        train_dataset = setup_dataset(cfg.data)
        num_train_samples = len(train_dataset)
        rank_0_info(f"Training dataset created with {num_train_samples} samples")
        rank_0_info(f"=" * 100)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        if cfg.data.eval_subset_size is not None:
            dataset_kwargs = {"max_samples": cfg.data.eval_subset_size}
        else:
            dataset_kwargs = {}

        eval_dataset = setup_dataset(cfg.data, is_eval=True, **dataset_kwargs)
        num_eval_samples = len(eval_dataset)
        rank_0_info(f"Evaluation dataset created with {num_eval_samples} samples")

    banner("Setting up trainer", f"Trainer class: {cfg.trainer_cls}")
    trainer_cls = {
        "rfm_heads": RFMHeadsTrainer,
        "rewind_transformer": ReWiNDTrainer,
        "rfm_vqa": RFMVQATrainer,
        "rewind_scale_transformer": ReWiNDTrainer,
    }[cfg.trainer_cls]

    # Add SaveBestCallback to automatically save and upload best models
    save_best_cfg = cfg.logging.save_best
    save_callback = SaveBestCallback(
        **asdict(save_best_cfg),
        base_model=cfg.model.base_model_id,
    )

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
    rank_0_info(f"🔧 DEBUG: Trainer callbacks: {[type(cb).__name__ for cb in trainer.callback_handler.callbacks]}")

    metrics_info = []
    for name, is_better in zip(save_best_cfg.metric_names, save_best_cfg.greater_is_better):
        direction = "↗️ higher" if is_better else "↘️ lower"
        metrics_info.append(f"{name} ({direction})")

    rank_0_info(f"💾 SaveBest monitoring: {', '.join(metrics_info)}")
    rank_0_info(f"📁 Keeping top {save_best_cfg.keep_top_k} checkpoint(s) and upload(s)")

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

    rank_0_info(f"Timing raw: {timing_raw}")

    checkpoint_path = resolve_checkpoint_path(cfg.training.resume_from_checkpoint, hub_token=save_best_cfg.hub_token)
    rank_0_info(f"Training from checkpoint: {checkpoint_path}")

    if cfg.debug:
        rank_0_info("🐛 DEBUG MODE: eval_steps=2, custom_eval_steps=2, eval_subset_size=10")

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model(cfg.training.output_dir)
    rank_0_info(f"Training complete! Check {cfg.training.output_dir} for checkpoints and final model.")


@hydra_main(version_base=None, config_path="rfm/configs", config_name="config")
def main(cfg: DictConfig):
    banner("Starting RFM Training")

    # Convert Hydra config to dataclass
    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    # Display the configuration in a nice Rich format
    display_config(exp_cfg)

    if exp_cfg.mode == "train":
        if is_rank_0():
            rprint(Panel.fit("🚀 Starting RFM Training", style="bold green"))
        train(exp_cfg)
    else:
        raise ValueError(f"Unknown mode: {exp_cfg.mode}. Must be 'train' or 'evaluate'")


if __name__ == "__main__":
    main()
