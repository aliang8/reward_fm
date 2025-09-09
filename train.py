import pyrallis
import warnings
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLModel,
    TrainingArguments,
)

from PIL import Image
import json
import os
import yaml
from rfm.trainers.rfm_heads_trainer import RFMHeadsTrainer
from rfm.trainers.vqa_trainer import VQATrainer
from rfm.trainers.rewind_trainer import ReWiNDTrainer
from rfm.utils.logging import is_rank_0, rank_0_print
from pyrallis import wrap
import wandb
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from dataclasses import asdict
import yaml

# Import shared configs and utilities
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.utils.setup_utils import (
    setup_model_and_processor,
    setup_peft_model,
    create_training_arguments,
    setup_data_generator,
    setup_batch_collator,
    setup_dataset,
    setup_eval_dataset,
    # setup_rewind_batch_collator,
    setup_vqa_model_and_processor,
    setup_vqa_batch_collator,
    # setup_transformer_model_and_processor
)
from rfm.utils.parser import parse_multiple
from rfm.utils.logging import _timer

# Suppress FSDP ShardedTensor deprecation warning
warnings.filterwarnings("ignore", message="Please use DTensor instead and we are deprecating ShardedTensor")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(cfg: ExperimentConfig):
    timing_raw = {}
    # Create DataGenerator for training using shared utility
    with _timer("time/setup_data_generator", timing_raw=timing_raw):
        data_generator = setup_data_generator(cfg)

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
    peft_rfm_model = setup_peft_model(rfm_model, cfg)

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

    # Use the shared utilities for batch collator and dataset
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_batch_collator(processor, tokenizer, cfg)
        train_dataset = setup_dataset(data_generator)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        eval_dataset = setup_eval_dataset(cfg)
        rank_0_print(f"Evaluation dataset created with {cfg.data.eval_subset_size} samples")

    trainer_cls = {
        "rfm_heads": RFMHeadsTrainer,
        "rewind_transformer": ReWiNDTrainer,
        "rfm_vqa": VQATrainer,
    }[cfg.trainer_cls]

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
            print(f"Trainer's Accelerator object found.")
            fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin:
                print(f"FSDP Plugin found in Accelerator state.")
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


def train_vqa(cfg: ExperimentConfig):
    # If cfg is a string (config path), load it
    if isinstance(cfg, str):
        cfg = ExperimentConfig.from_yaml(cfg)

    timing_raw = {}
    # Create DataGenerator for training using shared utility
    with _timer("time/setup_data_generator", timing_raw=timing_raw):
        data_generator = setup_data_generator(cfg)

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
        processor, vqa_model = setup_vqa_model_and_processor(cfg.model)

    # # Apply PEFT if enabled - Disabled for now
    # peft_vqa_model = setup_peft_model(vqa_model, cfg)

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

    # Use the shared utilities for batch collator and dataset
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_vqa_batch_collator(processor, cfg)
        train_dataset = setup_dataset(data_generator)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        eval_dataset = setup_eval_dataset(cfg)
        rank_0_print(f"Evaluation dataset created with {cfg.data.eval_subset_size} samples")

    trainer = VQATrainer(
        model=vqa_model,
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
            print(f"Trainer's Accelerator object found.")
            fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin:
                print(f"FSDP Plugin found in Accelerator state.")
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

def train_transformer(cfg: ExperimentConfig):
    timing_raw = {}
    # Create DataGenerator for training using shared utility
    with _timer("time/setup_data_generator", timing_raw=timing_raw):
        data_generator = setup_data_generator(cfg) # should be same as RFM the data format will be same
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
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw): # here need to change to custom transformer
        tokenizer, processor, rewind_model = setup_transformer_model_and_processor(cfg.model)
    train_args = create_training_arguments(cfg, cfg.training.output_dir)
    # Save config to output directory
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.training.output_dir, "transformer_config.yaml")
    config_dict = asdict(cfg)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    rank_0_print(f"Saved training config to: {config_save_path}")
    # Use the shared utilities for batch collator and dataset
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_rewind_batch_collator(processor, tokenizer, cfg)
        print("batch_collator is finished", batch_collator)
        train_dataset = setup_dataset(data_generator)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        eval_dataset = setup_eval_dataset(cfg)
        rank_0_print(f"Evaluation dataset created with {cfg.data.eval_subset_size} samples")

    trainer = ReWiNDTrainer(
        model=rewind_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=batch_collator,
        config=cfg,
    )

    if cfg.debug:
        cfg.training.save_steps = 2
        cfg.training.logging_steps = 2
        cfg.training.eval_steps = 2
        cfg.data.eval_subset_size = 10

    # if is_rank_0():
    #     print("\n" + "=" * 80)
    #     print("--- PRE-TRAINING FSDP DIAGNOSTICS ---")
    #     # The Trainer creates its own Accelerator instance. Let's check its state.
    #     if hasattr(trainer, "accelerator"):
    #         print(f"Trainer's Accelerator object found.")
    #         fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    #         if fsdp_plugin:
    #             print(f"FSDP Plugin found in Accelerator state.")
    #             # This is the configuration the accelerator will ACTUALLY use for wrapping.
    #             print(f"VERIFY: Actual FSDP plugin config being used: {fsdp_plugin}")
    #         else:
    #             print("ERROR: FSDP Plugin NOT found in the Trainer's accelerator state!")
    #     else:
    #         print("ERROR: Trainer has no 'accelerator' attribute yet. This check needs to be later.")
    #     print("=" * 80 + "\n")

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

    # Print a nice header
    console.print(Panel.fit("🤖 RFM (Reward Function Model) Configuration", style="bold cyan"))

    # Create a table for the main config
    table = Table(title="Experiment Settings", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="green")
    table.add_column("Value", style="yellow")

    # Mode and debug
    table.add_row("General", "Mode", cfg.mode)
    table.add_row("General", "Debug", str(cfg.debug))

    # Model config
    table.add_row("Model", "Base Model", cfg.model.base_model_id)
    table.add_row("Model", "Torch Dtype", cfg.model.torch_dtype)
    table.add_row("Model", "Trust Remote Code", str(cfg.model.trust_remote_code))

    # Data config
    table.add_row("Data", "Training Datasets", ", ".join(cfg.data.train_datasets))
    table.add_row("Data", "Training Subsets", ", ".join(str(subset) for subset in cfg.data.train_subsets))
    table.add_row("Data", "Eval Datasets", ", ".join(cfg.data.eval_datasets))
    table.add_row("Data", "Eval Subsets", ", ".join(str(subset) for subset in cfg.data.eval_subsets))
    table.add_row("Data", "Max Frames", str(cfg.data.max_frames))
    table.add_row("Data", "Video Frame Sampling", cfg.data.video_frame_sampling)
    table.add_row("Data", "Resized Height", str(cfg.data.resized_height))
    table.add_row("Data", "Resized Width", str(cfg.data.resized_width))
    table.add_row("Data", "Preference Ratio", str(cfg.data.preference_ratio))
    table.add_row("Data", "Dataset Preference Ratio", str(cfg.data.dataset_preference_ratio))
    table.add_row("Data", "Shuffle", str(cfg.data.shuffle))
    table.add_row("Data", "Seed", str(cfg.data.seed))
    table.add_row("Data", "Num Proc", str(cfg.data.num_proc))
    table.add_row("Data", "Force Reprocess", str(cfg.data.force_reprocess))
    table.add_row("Data", "Dataloader Pin Memory", str(cfg.data.dataloader_pin_memory))
    table.add_row("Data", "Dataloader Num Workers", str(cfg.data.dataloader_num_workers))

    # Training config
    table.add_row("Training", "Number of GPUs", str(cfg.training.num_gpus))
    table.add_row("Training", "Output Directory", cfg.training.output_dir)
    table.add_row("Training", "Batch Size", str(cfg.training.per_device_train_batch_size))
    table.add_row("Training", "Learning Rate", f"{cfg.training.learning_rate:.2e}")
    table.add_row("Training", "Epochs", str(cfg.training.num_train_epochs))
    table.add_row("Training", "Max Seq Length", str(cfg.training.max_seq_length))
    table.add_row("Training", "Beta", str(cfg.training.beta))
    table.add_row("Training", "Gradient Accumulation", str(cfg.training.gradient_accumulation_steps))
    table.add_row("Training", "Save Strategy", cfg.training.save_strategy)
    table.add_row("Training", "Logging Steps", str(cfg.training.logging_steps))
    table.add_row("Training", "Save Steps", str(cfg.training.save_steps))
    table.add_row("Training", "FP16", str(cfg.training.fp16))
    table.add_row("Training", "BF16", str(cfg.training.bf16))

    # PEFT config
    table.add_row("PEFT", "Use PEFT", str(cfg.peft.use_peft))
    if cfg.peft.use_peft:
        table.add_row("PEFT", "LoRA Rank (r)", str(cfg.peft.r))
        table.add_row("PEFT", "LoRA Alpha", str(cfg.peft.lora_alpha))
        table.add_row("PEFT", "LoRA Dropout", str(cfg.peft.lora_dropout))
        table.add_row("PEFT", "Target Modules", ", ".join(cfg.peft.target_modules))
        table.add_row("PEFT", "Train Vision Encoder", str(cfg.model.train_vision_encoder))
        table.add_row("PEFT", "Train Language Model", str(cfg.model.train_language_model))
        table.add_row("PEFT", "Train Value Head", str(cfg.model.train_value_head))
        table.add_row("PEFT", "Train Progress Head", str(cfg.model.train_progress_head))
        table.add_row("PEFT", "Train Preference Head", str(cfg.model.train_preference_head))
        table.add_row("PEFT", "Train Similarity Head", str(cfg.model.train_similarity_head))

    # Logging config
    table.add_row("Logging", "Use Wandb", str(cfg.logging.use_wandb))
    if cfg.logging.use_wandb:
        table.add_row("Logging", "Wandb Project", cfg.logging.wandb_project)
        table.add_row("Logging", "Wandb Entity", str(cfg.logging.wandb_entity))
        table.add_row("Logging", "Wandb Run Name", str(cfg.logging.wandb_run_name))

    console.print(table)

def main(cfg: ExperimentConfig):    
    # Display the configuration in a nice Rich format
    display_config(cfg)
    assert cfg.data.model_type in ["default", "vqa"], "Model type must be either 'default' or 'vqa'"

    if cfg.mode == "train" and cfg.data.model_type == "default": # and "Qwen" in cfg.model.base_model_id:
        if is_rank_0():
            rprint(Panel.fit("🚀 Starting RFM Training", style="bold green"))
        train(cfg)
    # elif cfg.mode == "train" and cfg.data.model_type == "vqa":  # VQA training
    #     if is_rank_0():
    #         rprint(Panel.fit("🧠 Starting VQA Baseline Training", style="bold cyan"))
    #     train_vqa(cfg)
    # elif cfg.mode == "train" and cfg.data.model_type == "default" and "Qwen" not in cfg.model.base_model_id:
    #     # train rewind transformer
    #     if is_rank_0():
    #         rprint(Panel.fit("🔄 Starting ReWiND Transformer Training", style="bold magenta"))
    #     train_transformer(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'train' or 'evaluate'")

if __name__ == "__main__":
    cfg = parse_multiple(ExperimentConfig)
    main(cfg)
