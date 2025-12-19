import collections
import copy
import json
import os
import random
from typing import Dict

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer

from rfm.data.datasets.base import resolve_dataset_keys
from rfm.data.datasets.helpers import load_frames_from_npz
from rfm.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING
from rfm.evals.compile_results import compute_eval_metrics
from rfm.evals.eval_metrics_utils import compute_pearson, compute_spearman
from rfm.models.utils import ModelOutput
from rfm.utils.distributed import banner, get_rank, is_rank_0, log_fsdp_diagnostics
from rfm.utils.logger import Logger, get_logger, log_memory_usage
from rfm.utils.metrics import compute_spearman_correlation
from rfm.utils.setup_utils import setup_batch_collator, setup_custom_eval_dataset, setup_dataset
from rfm.utils.tensor_utils import t2n
from rfm.utils.timer import _timer
from rfm.utils.video_utils import create_policy_ranking_grid, create_video_grid_with_progress
from PIL import Image, ImageDraw, ImageFont

logger = get_logger()


def seed_worker(worker_id):
    """Set random seed for dataloader workers."""
    import random

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reduce_metrics_with_accelerate(metrics: dict, accelerator, aggregate_method="sum"):
    """
    Reduce multiple scalar metrics using Accelerate's built-in methods.
    Handles cases where different processes have different metric keys.
    metrics: dict of {name: float or tensor}
    Returns dict with averaged metrics across all ranks.
    """
    if not metrics:
        return metrics

    # Step 1: Gather all metric keys from all processes
    local_keys = list(metrics.keys())
    all_keys_gathered = accelerator.gather_for_metrics(local_keys)

    # Step 2: Create union of all keys across all processes
    all_unique_keys = set()
    for keys_from_process in all_keys_gathered:
        if isinstance(keys_from_process, list):
            all_unique_keys.update(keys_from_process)
        else:
            # Handle single key case
            all_unique_keys.add(keys_from_process)

    all_unique_keys = sorted(all_unique_keys)

    # Step 3: Create synchronized metrics dict with 0.0 for missing keys
    synchronized_metrics = {}
    for key in all_unique_keys:
        if key in metrics:
            synchronized_metrics[key] = metrics[key]
        else:
            # This process doesn't have this metric, use 0.0
            synchronized_metrics[key] = 0.0

    # Step 4: Now reduce all metrics (all processes have same keys)
    result_metrics = {}

    for key, value in synchronized_metrics.items():
        try:
            # Convert to tensor on accelerator device
            if torch.is_tensor(value):
                tensor_val = value.to(accelerator.device, dtype=torch.float32)
            else:
                tensor_val = torch.tensor(float(value), dtype=torch.float32, device=accelerator.device)

            # Check for NaN values before reduction
            if torch.isnan(tensor_val).any():
                logger.warning(f"NaN detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Check for infinity values
            if torch.isinf(tensor_val).any():
                logger.warning(f"Infinity detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Use accelerator's reduce method - all processes participate
            reduced_val = accelerator.reduce(tensor_val, reduction=aggregate_method)

            # Final check for NaN in reduced result
            if torch.isnan(reduced_val).any():
                logger.warning(f"NaN in reduced result for metric '{key}', using fallback")
                result_metrics[key] = 0.0
            else:
                result_metrics[key] = reduced_val.item()

        except Exception as metric_error:
            # If individual metric fails, keep original value (or 0.0 if missing)
            logger.warning(f"Failed to reduce metric '{key}': {metric_error}")
            if key in metrics:
                original_val = float(metrics[key]) if not torch.is_tensor(metrics[key]) else metrics[key].item()
                result_metrics[key] = 0.0 if np.isnan(original_val) else original_val
            else:
                result_metrics[key] = 0.0

    # Step 5: Return all reduced metrics (all processes should have the same keys after reduction)
    # Return all keys from result_metrics to ensure we get all metrics across all processes
    return result_metrics


class RFMHeadsTrainer(Trainer):
    def __init__(self, config, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_metadata = collections.defaultdict(float)
        self.global_metadata = collections.defaultdict(float)
        self.timing_raw = collections.defaultdict(float)
        self._ddp_static_graph_set = False  # Flag to track if DDP static graph has been set
        self._fsdp_diagnostics_logged = False  # Flag to track if FSDP diagnostics have been logged

        if logger is not None:
            self.logger = logger
        else:
            log_level = self.config.logging.log_level
            self.logger = Logger(
                log_to=self.config.logging.log_to,
                output_dir=getattr(self.args, "output_dir", "./logs"),
                is_main_process=is_rank_0(),
                log_level=log_level,
            )

        # Use loguru logger after it's been initialized
        loguru_logger = get_logger()
        loguru_logger.info(f"DDP find_unused_parameters: {getattr(self.args, 'ddp_find_unused_parameters', 'N/A')}")

    def create_optimizer(self):
        """
        Override to create optimizer with separate parameter groups for vision encoder layers.
        If vision_encoder_lr is set, the last N vision encoder layers will use that LR,
        while all other parameters use the default learning rate.
        """
        # Check if we need to create parameter groups for vision encoder
        vision_encoder_lr = self.config.training.vision_encoder_lr
        vision_encoder_num_layers = self.config.training.vision_encoder_num_layers

        if vision_encoder_lr is None or vision_encoder_lr <= 0:
            # No special vision encoder LR, use default optimizer
            return super().create_optimizer()

        # Get the model
        model = self.model
        if not hasattr(model, "model") or not hasattr(model.model, "visual"):
            logger.warning(
                "vision_encoder_lr is set but model doesn't have visual encoder. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        # Get vision encoder blocks
        visual_encoder = model.model.visual
        if not hasattr(visual_encoder, "blocks"):
            logger.warning(
                "vision_encoder_lr is set but visual encoder doesn't have blocks. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        blocks = visual_encoder.blocks
        total_blocks = len(blocks)

        if vision_encoder_num_layers > total_blocks:
            logger.warning(
                f"vision_encoder_num_layers ({vision_encoder_num_layers}) is greater than "
                f"total blocks ({total_blocks}). Using all blocks for vision encoder LR."
            )
            vision_encoder_num_layers = total_blocks

        # Identify parameters for last N layers
        vision_encoder_params = []
        other_params = []

        # Get all parameters and their names
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this parameter belongs to the last N vision encoder blocks
            is_vision_encoder_param = False
            if "visual.blocks" in name:
                # Extract block index from parameter name
                # Format: model.visual.blocks.{idx}.{rest}
                try:
                    parts = name.split("visual.blocks.")
                    if len(parts) > 1:
                        block_part = parts[1].split(".")[0]
                        block_idx = int(block_part)
                        # Check if this is one of the last N blocks
                        if block_idx >= (total_blocks - vision_encoder_num_layers):
                            is_vision_encoder_param = True
                except (ValueError, IndexError):
                    # If we can't parse the block index, skip this parameter
                    pass

            if is_vision_encoder_param:
                vision_encoder_params.append(param)
            else:
                other_params.append(param)

        if not vision_encoder_params:
            logger.warning(
                "No vision encoder parameters found for parameter groups. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        # Use AdamW as default (same as HuggingFace Trainer)
        optimizer_kwargs = {
            "betas": (
                self.args.adam_beta1 if hasattr(self.args, "adam_beta1") else 0.9,
                self.args.adam_beta2 if hasattr(self.args, "adam_beta2") else 0.999,
            ),
            "eps": self.args.adam_epsilon if hasattr(self.args, "adam_epsilon") else 1e-8,
            "weight_decay": self.args.weight_decay,
        }

        # Create parameter groups with different learning rates
        param_groups = [
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                **optimizer_kwargs,
            },
            {
                "params": vision_encoder_params,
                "lr": vision_encoder_lr,
                **optimizer_kwargs,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)

        logger.info(
            f"Created optimizer with parameter groups: "
            f"{len(other_params)} params at LR={self.args.learning_rate}, "
            f"{len(vision_encoder_params)} vision encoder params (last {vision_encoder_num_layers} blocks) at LR={vision_encoder_lr}"
        )
        self.optimizer = optimizer

        return optimizer

    def _post_checkpoint_load_reset(self):
        """
        Reset model and optimizer state after loading from checkpoint.
        This addresses issues where checkpoint loading can leave stale gradients
        or computational graph state that causes crashes during training.
        """
        logger.info("Performing post-checkpoint load reset...")

        # Ensure model is in training mode
        self.model.train()

        # Clear any cached gradients or computational graph state
        # NOTE: We don't clear optimizer.state or param_groups as that breaks the lr_scheduler
        try:
            # Zero out any existing gradients
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            logger.warning(f"Could not clear gradients: {e}")

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Post-checkpoint load reset complete")

    def _normalize_list_like(self, value):
        """Convert None/scalars/tuples to a list so we can safely gather across ranks."""
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _gather_list_across_processes(self, value):
        """Gather Python lists (or list-like) across all ranks."""
        normalized = self._normalize_list_like(value)
        if not dist.is_initialized():
            return normalized

        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, normalized)

        flattened = []
        for proc_list in gathered:
            if not proc_list:
                continue
            flattened.extend(proc_list)
        return flattened

    def _gather_metadata_fields(self, sample_inputs: dict, fields: list[str]) -> dict:
        """Gather heterogeneous metadata fields (lists, tensors) across processes."""
        gathered = {}
        for field in fields:
            field_value = sample_inputs.get(field)
            if torch.is_tensor(field_value):
                gathered[field] = self.accelerator.gather_for_metrics(field_value)
            else:
                gathered[field] = self._gather_list_across_processes(field_value)
        return gathered

    def _truncate_metadata_lists(self, metadata: dict, max_len: int) -> dict:
        """Ensure metadata lists align with tensor batch sizes without altering tensors."""
        if max_len < 0:
            return metadata
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = value[:max_len]
        return metadata

    def _get_learning_rate(self):
        """
        Override to safely get learning rate, handling cases where scheduler hasn't been stepped yet.
        """
        try:
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                last_lrs = self.lr_scheduler.get_last_lr()
                if last_lrs:
                    return last_lrs[0]
            # Fallback to optimizer's learning rate
            if hasattr(self, "optimizer") and self.optimizer is not None:
                if self.optimizer.param_groups:
                    return self.optimizer.param_groups[0]["lr"]
            # Last resort: return configured learning rate
            return self.args.learning_rate
        except Exception as e:
            logger.warning(f"Could not get learning rate: {e}")
            return self.args.learning_rate

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Override train method to perform post-checkpoint reset.
        """
        # If resuming from checkpoint, set flag for reset in first training step
        if resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self._just_resumed_from_checkpoint = True

        # Call parent train method
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        logger.debug("training_step: Starting")

        if not self._fsdp_diagnostics_logged:
            log_fsdp_diagnostics(model, accelerator=self.accelerator, logger=logger)
            self._fsdp_diagnostics_logged = True

        # Check if we just resumed from checkpoint (first step after resume)
        if hasattr(self, "_just_resumed_from_checkpoint") and self._just_resumed_from_checkpoint:
            self._post_checkpoint_load_reset()
            self._just_resumed_from_checkpoint = False

        self.timing_raw = {}

        # Initialize log_metadata
        self.log_metadata = {}

        # Safety check: ensure model is in training mode and gradients are properly set up
        if not model.training:
            logger.warning("Model not in training mode, setting to train mode")
            model.train()

        # Clear any stale gradients before starting
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        with _timer("time/training_step", timing_raw=self.timing_raw):
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_progress = inputs.get("num_progress", 0)
        num_similarities = inputs.get("num_similarities", 0)

        logger.trace(
            f"num_preferences: {num_preferences}, num_progress: {num_progress}, num_similarities: {num_similarities}"
        )

        if num_preferences > 0 and preference_inputs:
            rejected_data_gen_strategy = preference_inputs["rejected_data_gen_strategy"]
            if isinstance(rejected_data_gen_strategy, list) and len(rejected_data_gen_strategy) > 0:
                for s in rejected_data_gen_strategy:
                    self.global_metadata[f"pref_{s}"] += 1

            data_sources = preference_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        if num_progress > 0 and progress_inputs:
            data_gen_strategy = progress_inputs["data_gen_strategy"]
            if isinstance(data_gen_strategy, list) and len(data_gen_strategy) > 0:
                for s in data_gen_strategy:
                    self.global_metadata[f"prog_{s}"] += 1

            data_sources = progress_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        if num_similarities > 0 and similarity_inputs:
            data_gen_strategy = similarity_inputs["data_gen_strategy"]
            if isinstance(data_gen_strategy, list) and len(data_gen_strategy) > 0:
                for s in data_gen_strategy:
                    self.global_metadata[f"sim_{s}"] += 1

            data_sources = similarity_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        # Update global metadata for training
        # add to total batch size and sum across all processes
        self.global_metadata["total_samples"] += num_preferences + num_similarities + num_progress
        self.global_metadata["total_preferences"] += num_preferences
        self.global_metadata["total_similarities"] += num_similarities
        self.global_metadata["total_progress"] += num_progress

        logger.trace("finished updating global metadata")

        # self._update_resample_attempt_metrics(inputs)

        # logger.trace("update resample attempt metrics")

        # Log custom losses at specified intervals (using our custom logger only)
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_metadata()

        # Log GPU memory usage at every training step for diagnostics
        log_memory_usage(f"Step {self.state.global_step}")

        return loss

    def _get_optimizer_stats(self):
        """Get optimizer and gradient statistics for logging."""
        optim_stats = {}

        if not hasattr(self, "optimizer") or self.optimizer is None:
            return optim_stats

        # Get learning rates for each parameter group
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group.get("lr", 0.0)
            optim_stats[f"optim/lr_group_{i}"] = lr

        # If only one param group, also log as optim/lr for convenience
        if len(self.optimizer.param_groups) == 1:
            optim_stats["optim/lr"] = self.optimizer.param_groups[0].get("lr", 0.0)

        # Compute gradient norms across all model parameters
        total_norm = 0.0
        num_params_with_grad = 0
        max_grad_norm = 0.0
        min_grad_norm = float("inf")

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                num_params_with_grad += 1
                max_grad_norm = max(max_grad_norm, param_norm)
                min_grad_norm = min(min_grad_norm, param_norm)

        if num_params_with_grad > 0:
            total_norm = total_norm**0.5
            optim_stats["optim/preclip_grad_norm"] = total_norm
            optim_stats["optim/preclip_grad_norm_max"] = max_grad_norm
            optim_stats["optim/preclip_grad_norm_min"] = min_grad_norm if min_grad_norm != float("inf") else 0.0
            optim_stats["optim/num_params_with_grad"] = num_params_with_grad

        # Compute parameter norms across all model parameters
        total_param_norm = 0.0
        max_param_norm = 0.0
        min_param_norm = float("inf")
        param_norms = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.data is None:
                continue
            param_norm = p.data.norm(2).item()
            total_param_norm += param_norm**2
            max_param_norm = max(max_param_norm, param_norm)
            min_param_norm = min(min_param_norm, param_norm)
            param_norms.append((name, param_norm))

        if param_norms:
            total_param_norm = total_param_norm**0.5
            optim_stats["optim/param_norm"] = total_param_norm
            optim_stats["optim/param_norm_max"] = max_param_norm
            optim_stats["optim/param_norm_min"] = min_param_norm if min_param_norm != float("inf") else 0.0

        # Get optimizer state statistics (e.g., momentum, variance for Adam)
        if hasattr(self.optimizer, "state") and len(self.optimizer.state) > 0:
            # For Adam-like optimizers, log average momentum and variance
            exp_avg_norms = []
            exp_avg_sq_norms = []

            for state in self.optimizer.state.values():
                if "exp_avg" in state:
                    exp_avg_norms.append(state["exp_avg"].norm(2).item())
                if "exp_avg_sq" in state:
                    exp_avg_sq_norms.append(state["exp_avg_sq"].norm(2).item())

            if exp_avg_norms:
                optim_stats["optim/exp_avg_norm_mean"] = np.mean(exp_avg_norms)
                optim_stats["optim/exp_avg_norm_max"] = np.max(exp_avg_norms)
            if exp_avg_sq_norms:
                optim_stats["optim/exp_avg_sq_norm_mean"] = np.mean(exp_avg_sq_norms)
                optim_stats["optim/exp_avg_sq_norm_max"] = np.max(exp_avg_sq_norms)

        # Log top 10 parameters with largest gradient norms
        param_grad_norms = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            grad_norm = p.grad.data.norm(2).item()
            param_grad_norms.append((name, grad_norm))

        if param_grad_norms:
            # Sort by gradient norm (descending) and take top 10
            param_grad_norms.sort(key=lambda x: x[1], reverse=True)
            for i, (name, grad_norm) in enumerate(param_grad_norms[:5]):
                # Shorten parameter name for cleaner logging
                short_name = name.replace("model.", "").replace("module.", "")
                optim_stats[f"optim/top_preclip_grad_norm_{i + 1}_{short_name}"] = grad_norm

        if param_norms:
            # Sort by parameter norm (descending) and take top 10
            param_norms.sort(key=lambda x: x[1], reverse=True)
            for i, (name, param_norm) in enumerate(param_norms[:5]):
                short_name = name.replace("model.", "").replace("module.", "")
                optim_stats[f"optim/top_param_norm_{i + 1}_{short_name}"] = param_norm

        return optim_stats

    def _update_resample_attempt_metrics(self, inputs: dict) -> None:
        """Aggregate resample attempt statistics across processes."""
        if not hasattr(self, "accelerator"):
            return

        local_pairs: list[tuple[str, float]] = []

        for key in ("preference_inputs", "progress_inputs", "similarity_inputs"):
            sample_inputs = inputs.get(key) or {}
            resample_attempts = sample_inputs.get("resample_attempts")
            if resample_attempts is None:
                continue

            if torch.is_tensor(resample_attempts):
                attempts_tensor = resample_attempts.to(self.accelerator.device, dtype=torch.float32).view(-1)
            else:
                attempts_tensor = torch.tensor(
                    resample_attempts, dtype=torch.float32, device=self.accelerator.device
                ).view(-1)

            if attempts_tensor.numel() == 0:
                continue

            sample_category = key.replace("_inputs", "")
            strategies = sample_inputs.get("data_gen_strategy")
            if strategies is None:
                raise ValueError(
                    f"Expected data_gen_strategy for {sample_category} samples when logging resample attempts."
                )

            if len(strategies) != attempts_tensor.numel():
                raise ValueError(
                    f"Mismatch between resample attempts ({attempts_tensor.numel()}) and strategies "
                    f"({len(strategies)}) for {sample_category} samples."
                )

            strategy_labels = [f"{sample_category}/{str(strategy)}" for strategy in strategies]

            for attempt_value, strategy_label in zip(attempts_tensor.tolist(), strategy_labels):
                local_pairs.append((strategy_label, float(attempt_value)))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_lists: list[list[tuple[str, float]]] = [None] * world_size
            dist.all_gather_object(gathered_lists, local_pairs)
            flat_pairs = [pair for proc_pairs in gathered_lists for pair in proc_pairs]
        else:
            flat_pairs = local_pairs

        if not flat_pairs:
            return

        all_attempts = [attempt for _, attempt in flat_pairs]
        self.log_metadata["data/resample_min"] = float(min(all_attempts))
        self.log_metadata["data/resample_max"] = float(max(all_attempts))
        self.log_metadata["data/resample_mean"] = float(sum(all_attempts) / len(all_attempts))

        strategy_values: dict[str, list[float]] = collections.defaultdict(list)
        for label, attempt in flat_pairs:
            strategy_values[label].append(attempt)

        for label, values in strategy_values.items():
            if not values:
                continue
            safe_label = label.replace("/", "_").replace(" ", "_")
            strategy_min = float(min(values))
            strategy_max = float(max(values))
            strategy_mean = float(sum(values) / len(values))
            self.log_metadata[f"data/resample_min_{safe_label}"] = strategy_min
            self.log_metadata[f"data/resample_max_{safe_label}"] = strategy_max
            self.log_metadata[f"data/resample_mean_{safe_label}"] = strategy_mean

    def _log_metadata(self):
        """Log custom RFM losses to wandb and console."""
        if not self.log_metadata:
            return

        logger.trace("logging metadata, starting to aggregate metrics")

        # Use local metrics (no aggregation needed for individual GPU metrics)
        log_metadata = reduce_metrics_with_accelerate(self.log_metadata, self.accelerator, aggregate_method="mean")

        logger.trace("finished aggregating metrics")

        training_step_time = self.timing_raw.get("time/training_step", 0.0)
        it_per_sec = 1.0 / training_step_time if training_step_time > 0 else 0.0

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "train/it_per_sec": it_per_sec,
            **self.timing_raw,
            **log_metadata,
        }

        # Log global metadata
        logger.trace("logging global metadata")
        global_metadata = reduce_metrics_with_accelerate(self.global_metadata, self.accelerator, aggregate_method="sum")
        logger.trace("finished aggregating global metadata")

        # Convert counts to fractions of total samples
        total_samples = global_metadata["total_samples"]
        log_global = {
            f"counts/{key}": value / total_samples for key, value in global_metadata.items() if key != "total_samples"
        }

        log_data.update(log_global)

        # Log optimizer and gradient statistics
        optim_stats = self._get_optimizer_stats()
        log_data.update(optim_stats)

        # make sure values are floats so they are loggable into wandb reports
        log_data = {k: float(v) for k, v in log_data.items()}

        self.logger.log_scalars(log_data, step=self.state.global_step)

        if is_rank_0():
            logger.info(f"Step {self.state.global_step}, Epoch {self.state.epoch:.2f}:")
            logger.info("-" * 50)
            logger.info(f"  train/it_per_sec: {it_per_sec:.4f}")
            for key in log_global:
                logger.info(f"  {key}: {log_global[key]}")

            rounded_times = {k: round(v, 2) for k, v in self.timing_raw.items()}
            logger.info(f"Timing raw: {rounded_times}")

            # Log optimizer stats to console
            if optim_stats:
                logger.info(f"Optimizer stats: {optim_stats}")

    def _make_eval_dataloader(self, dataset):
        """Create a distributed evaluation dataloader with proper sampling."""
        collator = setup_batch_collator(self.model.processor, self.model.tokenizer, self.config, is_eval=True)

        dl = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            # Force persistent_workers=False for eval to prevent memory leaks across datasets
            persistent_workers=False,
            worker_init_fn=seed_worker,
        )
        prepared_dl = self.accelerator.prepare(dl)
        return prepared_dl

    def _setup_eval_dataset(self, eval_type, eval_dataset):
        """Setup dataset and dataloader for evaluation."""
        eval_cfg = copy.deepcopy(self.config.data)
        eval_cfg.dataset_type = "rfm"
        # For similarity_score, eval_dataset is a list of datasets that should be loaded together
        # For other eval types, eval_dataset is a single dataset name
        if eval_type == "similarity_score" and isinstance(eval_dataset, list):
            eval_cfg.eval_datasets = eval_dataset
        else:
            eval_cfg.eval_datasets = [eval_dataset]

        # Create custom eval dataset with the appropriate sampler
        # set max_trajectories to 10 for reward_alignment per eval dataset
        kwargs = {}
        if eval_type == "reward_alignment":
            kwargs["max_trajectories"] = 10
            kwargs["frame_step"] = (
                2 if (self.config.trainer_cls == "rfm_heads" and not self.config.data.use_multi_image) else 1
            )
        if eval_type == "quality_preference":
            kwargs["comparisons_per_task"] = self.config.custom_eval.comparisons_per_task
        if eval_type == "policy_ranking":
            kwargs["num_examples_per_quality_pr"] = self.config.custom_eval.num_examples_per_quality_pr

        dataset = setup_custom_eval_dataset(eval_cfg, sampler_type=eval_type, is_eval=True, verbose=False, **kwargs)
        # Explicitly delete eval_cfg after dataset creation to free memory
        del eval_cfg

        logger.info(f"  Dataset size: {len(dataset)}")
        # log_memory_usage(f"After creating dataset")

        dataloader = self._make_eval_dataloader(dataset)
        logger.info(f"  Dataloader created with {len(dataloader)} batches")
        # log_memory_usage(f"After creating dataloader")

        # Ensure model is in eval mode and clear any gradient buffers
        self.model.eval()
        # Explicitly zero any gradients that might exist (shouldn't, but safety measure)
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Clear cache before starting evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # log_memory_usage(f"After clearing cache, before eval loop")

        return dataset, dataloader

    def _process_batch_progress_eval(self, batch, eval_type):
        """Process a batch for progress-based evaluations (reward_alignment, policy_ranking, confusion_matrix)."""
        logger.trace(f"    Processing {eval_type} batch")
        progress_samples = batch["progress_inputs"]
        logger.trace(f"    Calling forward_model for progress")
        with torch.no_grad():
            outputs, _ = self.forward_model(self.model, progress_samples, sample_type="progress")
        logger.trace(f"    Forward pass complete")

        progress_logits = outputs.progress_logits
        progress_pred = progress_logits["A"]

        # Gather everything
        progress_pred = self.accelerator.gather_for_metrics(progress_pred)
        target_progress = self.accelerator.gather_for_metrics(progress_samples["target_progress"])

        # Gather metadata fields
        metadata_fields = [
            "task",
            "data_source",
            "data_gen_strategy",
            "quality_labels",
            "metadata",
            "partial_success",
        ]
        gathered_metadata_dict = self._gather_metadata_fields(progress_samples, metadata_fields)
        num_progress_samples = progress_pred.shape[0] if progress_pred is not None else 0
        gathered_metadata_dict = self._truncate_metadata_lists(gathered_metadata_dict, num_progress_samples)

        # Handle success predictions if needed
        success_pred_gathered = None
        success_probs_gathered = None
        success_labels_gathered = None
        if self.config.model.train_success_head:
            success_pred = outputs.success_logits["A"]
            success_probs = torch.sigmoid(success_pred)
            success_binary = (success_probs > 0.5).float()
            success_pred_gathered = self.accelerator.gather_for_metrics(success_binary)
            success_probs_gathered = self.accelerator.gather_for_metrics(success_probs)
            success_labels = progress_samples.get("success_labels")
            success_labels_gathered = self.accelerator.gather_for_metrics(success_labels)

            # Clean up intermediate tensors (but keep gathered tensors for eval_results)
            del (
                success_pred,
                success_binary,
                success_probs,
                success_labels,
            )

        # Build eval_results on all processes for compute_eval_metrics
        batch_results = []
        for i in range(len(progress_pred)):
            metadata = gathered_metadata_dict["metadata"][i]
            sample_result = {
                "task": gathered_metadata_dict["task"][i],
                "target_progress": t2n(target_progress[i]),
                "progress_pred": t2n(progress_pred[i]),
                "data_source": gathered_metadata_dict["data_source"][i],
                "data_gen_strategy": gathered_metadata_dict["data_gen_strategy"][i],
                "quality_label": gathered_metadata_dict["quality_labels"][i],
                "metadata": metadata,
                "id": metadata["id"],
                "video_path": metadata["video_path"],
                "partial_success": gathered_metadata_dict["partial_success"][i],
            }
            if success_pred_gathered is not None:
                sample_result["success_pred"] = t2n(success_pred_gathered[i])
            if success_probs_gathered is not None:
                sample_result["success_probs"] = t2n(success_probs_gathered[i])
            if success_labels_gathered is not None:
                sample_result["success_labels"] = t2n(success_labels_gathered[i])
            batch_results.append(sample_result)

        # Clean up gathered tensors and metadata after building results
        del progress_pred, target_progress, gathered_metadata_dict
        if success_pred_gathered is not None:
            del success_pred_gathered
        if success_probs_gathered is not None:
            del success_probs_gathered
        if success_labels_gathered is not None:
            del success_labels_gathered

        return batch_results, outputs

    def _process_batch_preference_eval(self, batch):
        """Process a batch for preference-based evaluations (quality_preference)."""
        logger.trace(f"    Processing quality_preference batch")
        preference_samples = batch["preference_inputs"]
        logger.trace(f"    Calling forward_model for preference")
        with torch.no_grad():
            outputs, _ = self.forward_model(self.model, preference_samples, sample_type="preference")
        logger.trace(f"    Forward pass complete")
        pref_logits = outputs.pref_logits

        # Gather predictions and labels across all ranks
        pref_logits = self.accelerator.gather_for_metrics(pref_logits)
        preference_labels = self.accelerator.gather_for_metrics(preference_samples["preference_labels"])

        # Gather non-tensor metadata using helper (handles single and multi GPU)
        gathered_pref_metadata = self._gather_metadata_fields(
            preference_samples,
            [
                "task",
                "data_source",
                "chosen_data_gen_strategy",
                "rejected_data_gen_strategy",
                "metadata",
            ],
        )
        num_pref_samples = pref_logits.shape[0] if pref_logits is not None else 0
        gathered_pref_metadata = self._truncate_metadata_lists(gathered_pref_metadata, num_pref_samples)
        gathered_task = gathered_pref_metadata["task"]
        gathered_data_source = gathered_pref_metadata["data_source"]
        gathered_chosen_data_gen_strategy = gathered_pref_metadata["chosen_data_gen_strategy"]
        gathered_rejected_data_gen_strategy = gathered_pref_metadata["rejected_data_gen_strategy"]
        gathered_metadata = gathered_pref_metadata["metadata"]

        # Build eval_results on all processes for compute_eval_metrics
        batch_results = []
        for i in range(len(pref_logits)):
            if pref_logits[i] is None:
                continue
            sample_result = {
                "task": gathered_task[i],
                "preference_pred": t2n(pref_logits[i]),
                "preference_labels": t2n(preference_labels[i]),
                "data_source": gathered_data_source[i],
                "chosen_data_gen_strategy": gathered_chosen_data_gen_strategy[i],
                "rejected_data_gen_strategy": gathered_rejected_data_gen_strategy[i],
                "metadata": gathered_metadata[i],
            }
            batch_results.append(sample_result)

        # Clean up gathered tensors and metadata after building results
        del pref_logits, preference_labels
        del gathered_task, gathered_data_source, gathered_chosen_data_gen_strategy
        del gathered_rejected_data_gen_strategy, gathered_metadata

        return batch_results, outputs

    def _process_batch_similarity_eval(self, batch):
        """Process a batch for similarity-based evaluations (similarity_score)."""
        logger.trace(f"    Processing similarity_score batch")
        similarity_samples = batch["similarity_inputs"]

        # Log similarity batch details
        num_sim_samples = len(similarity_samples.get("data_source", []))
        logger.trace(f"    Similarity samples on this rank: {num_sim_samples}")
        if "input_ids" in similarity_samples:
            logger.trace(f"    input_ids shape: {similarity_samples['input_ids'].shape}")

        logger.trace(f"    Calling forward_model for similarity")
        # log_memory_usage(f"Before similarity forward pass")

        with torch.no_grad():
            outputs, _ = self.forward_model(self.model, similarity_samples, sample_type="similarity")

        logger.trace(f"    Forward pass complete")
        # log_memory_usage(f"After similarity forward pass")

        sim_logits = outputs.sim_logits
        logger.trace(f"    sim_logits shape: {sim_logits.shape if sim_logits is not None else 'None'}")

        # Gather predictions across all ranks
        logger.trace(f"    Gathering sim_logits across ranks")
        sim_logits = self.accelerator.gather_for_metrics(sim_logits)
        logger.trace(f"    Gathered sim_logits shape: {sim_logits.shape if sim_logits is not None else 'None'}")

        # Gather non-tensor metadata using helper (handles optional/None entries)
        logger.trace(f"    Gathering metadata across ranks")
        gathered_sim_metadata = self._gather_metadata_fields(
            similarity_samples,
            ["task", "data_source", "data_gen_strategy", "metadata"],
        )
        logger.trace(f"    Metadata gathered, building eval_results")
        num_sim_samples = len(sim_logits) // 2 if sim_logits is not None else 0
        gathered_sim_metadata = self._truncate_metadata_lists(gathered_sim_metadata, num_sim_samples)
        gathered_task = gathered_sim_metadata["task"]
        gathered_data_source = gathered_sim_metadata["data_source"]
        gathered_data_gen_strategy = gathered_sim_metadata["data_gen_strategy"]
        gathered_metadata = gathered_sim_metadata["metadata"]

        # Build eval_results on all processes for compute_eval_metrics
        # The sim_logits are batched as [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        # We need to extract ref_sim (even indices) and ref_diff (odd indices)
        # The metadata lists have length = num_samples, but sim_logits has length = 2 * num_samples
        batch_results = []
        num_samples = len(sim_logits) // 2
        for i in range(num_samples):
            ref_sim_idx = i * 2
            ref_diff_idx = i * 2 + 1

            if ref_sim_idx >= len(sim_logits) or ref_diff_idx >= len(sim_logits):
                continue

            # Metadata is indexed by sample index (i), not batched index
            sample_result = {
                "task": gathered_task[i] if i < len(gathered_task) else None,
                "sim_score_ref_sim": t2n(sim_logits[ref_sim_idx]),
                "sim_score_ref_diff": t2n(sim_logits[ref_diff_idx]),
                "data_source": gathered_data_source[i] if i < len(gathered_data_source) else None,
                "data_gen_strategy": gathered_data_gen_strategy[i] if i < len(gathered_data_gen_strategy) else None,
                "metadata": gathered_metadata[i] if i < len(gathered_metadata) else None,
            }
            batch_results.append(sample_result)

        # Clean up gathered tensors and metadata after building results
        del sim_logits
        del gathered_task, gathered_data_source, gathered_data_gen_strategy, gathered_metadata

        return batch_results, outputs

    def _save_reward_alignment_videos(
        self, video_frames_list, plots, eval_results, output_dir, ds_name, trajectory_progress_data=None
    ):
        # Check if dataset is RoboArena
        is_roboarena = False
        if eval_results and len(eval_results) > 0:
            first_data_source = eval_results[0].get("data_source", "")
            is_roboarena = "roboarena" in str(first_data_source).lower()

        # Group eval_results by trajectory ID (like compile_results.py does)
        # Since compile_results processes trajectories in order and creates video_frames_list/plots
        # in that same order, we can reconstruct the mapping by collecting unique trajectory IDs
        processed_trajectory_ids = []
        for r in eval_results:
            trajectory_id = r.get("id")
            if trajectory_id and trajectory_id not in processed_trajectory_ids:
                processed_trajectory_ids.append(trajectory_id)

        saved_count = 0
        for idx, (frames, plot) in enumerate(zip(video_frames_list, plots)):
            if frames is None or idx >= len(processed_trajectory_ids):
                continue

            trajectory_id = processed_trajectory_ids[idx]
            # Get all results for this trajectory
            results_for_trajectory = [r for r in eval_results if r.get("id") == trajectory_id]
            results_for_trajectory.sort(key=lambda r: r.get("metadata", {}).get("subsequence_end", 0))

            if not results_for_trajectory:
                continue

            result = results_for_trajectory[0]
            data_source = result.get("data_source", "unknown")
            quality_label = result.get("quality_label", "unknown")
            traj_id = result.get("id", f"traj_{idx}")
            partial_success = result.get("partial_success")
            task = result.get("task", "unknown")

            # Build directory structure: {data_source}/{quality_label}/
            save_dir = os.path.join(output_dir, "reward_alignment_videos", str(data_source), str(quality_label))
            os.makedirs(save_dir, exist_ok=True)

            # Build filename
            if is_roboarena and partial_success is not None:
                # For RoboArena: {quality_label}_{partial_success}_{traj_id}.mp4
                filename = f"{quality_label}_{partial_success}_{traj_id}.mp4"
            else:
                # Standard: {traj_id}.mp4
                filename = f"{traj_id}.mp4"

            video_path = os.path.join(save_dir, filename)

            # Load original frames at full resolution from video_path
            video_path_from_result = result.get("video_path")
            original_frames = load_frames_from_npz(video_path_from_result)
            # frames are in (T, H, W, C) format from load_frames_from_npz
            if original_frames.shape[-1] == 3:
                frames_rgb = original_frames.astype(np.uint8)
            elif original_frames.shape[1] == 3:
                # If in (T, C, H, W) format, transpose to (T, H, W, C)
                frames_rgb = original_frames.transpose(0, 2, 3, 1).astype(np.uint8)
            else:
                frames_rgb = original_frames.astype(np.uint8)

            # Ensure frames are in correct range [0, 255]
            if frames_rgb.max() <= 1.0:
                frames_rgb = (frames_rgb * 255).astype(np.uint8)
            else:
                frames_rgb = np.clip(frames_rgb, 0, 255).astype(np.uint8)

            # Get progress data for this trajectory (should always be available)
            progress_pred = None
            target_progress = None
            if trajectory_progress_data and idx < len(trajectory_progress_data):
                traj_data = trajectory_progress_data[idx]
                progress_pred = traj_data.get("progress_pred")
                target_progress = traj_data.get("target_progress")

            # If we don't have progress data from trajectory_progress_data, extract from results_for_trajectory
            if progress_pred is None or target_progress is None:
                progress_pred = []
                target_progress = []
                for r in results_for_trajectory:
                    pred = r.get("progress_pred")
                    tgt = r.get("target_progress")
                    if pred is not None:
                        # Use prediction at current timestep (or last if past max length)
                        timestep = len(progress_pred)
                        if timestep >= len(pred) - 1:
                            progress_pred.append(float(pred[-1]))
                        else:
                            progress_pred.append(float(pred[timestep]))
                    else:
                        progress_pred.append(0.0)

                    if tgt is not None and len(tgt) > 0:
                        target_progress.append(float(tgt[-1]))
                    else:
                        target_progress.append(0.0)

                # Handle relative progress type
                if self.config.data.progress_pred_type == "relative":
                    progress_pred = np.cumsum(np.array(progress_pred)).tolist()
                    target_progress = np.cumsum(np.array(target_progress)).tolist()

            # Get success data for this trajectory (if available)
            success_probs = None
            success_labels = None
            for r in results_for_trajectory:
                if r.get("success_probs") is not None:
                    if success_probs is None:
                        success_probs = []
                    sp = r.get("success_probs")
                    if sp is not None:
                        # Use probability at current timestep (or last if past max length)
                        timestep = len(success_probs)
                        if isinstance(sp, (list, np.ndarray)):
                            if timestep >= len(sp) - 1:
                                success_probs.append(float(sp[-1]))
                            else:
                                success_probs.append(float(sp[timestep]))
                        else:
                            success_probs.append(float(sp))
                    else:
                        success_probs.append(0.0)

                if r.get("success_labels") is not None:
                    if success_labels is None:
                        success_labels = []
                    sl = r.get("success_labels")
                    if sl is not None:
                        # Use label at current timestep (or last if past max length)
                        timestep = len(success_labels)
                        if isinstance(sl, (list, np.ndarray)):
                            if timestep >= len(sl) - 1:
                                success_labels.append(float(sl[-1]))
                            else:
                                success_labels.append(float(sl[timestep]))
                        else:
                            success_labels.append(float(sl))
                    else:
                        success_labels.append(0.0)

            # Ensure we have progress data (should always be available)
            if progress_pred is None or target_progress is None:
                logger.warning(f"No progress data available for trajectory {traj_id}, skipping video")
                continue

            # Create matplotlib animated plot video
            # Define DPI for animation quality
            fig_dpi = 300

            # Use matplotlib animation to save video with subplots
            try:
                # Compute metrics
                last_preds = np.array(progress_pred)
                last_targets = np.array(target_progress)

                # Check if this is a failure dataset
                from rfm.data.dataset_category import is_failure

                is_failure_dataset = is_failure(data_source)

                if is_failure_dataset:
                    traj_mse = 0.0
                    traj_pearson = 0.0
                    traj_spearman = 0.0
                else:
                    traj_mse = float(np.mean((last_targets - last_preds) ** 2))
                    traj_pearson = compute_pearson(last_targets.tolist(), last_preds.tolist())
                    traj_spearman = compute_spearman(last_targets.tolist(), last_preds.tolist())

                    # Handle NaN values
                    traj_pearson = float(traj_pearson) if not np.isnan(traj_pearson) else 0.0
                    traj_spearman = float(traj_spearman) if not np.isnan(traj_spearman) else 0.0

                # Determine number of subplots: 3 if success data available, 2 otherwise
                has_success_data = success_probs is not None and len(success_probs) > 0
                num_subplots = 3 if has_success_data else 2

                # Create figure with subplots: video, progress plot, and optionally success plot
                if has_success_data:
                    fig_anim, (ax_video, ax_progress, ax_success) = plt.subplots(1, 3, figsize=(24, 8), dpi=fig_dpi)
                else:
                    fig_anim, (ax_video, ax_progress) = plt.subplots(1, 2, figsize=(16, 8), dpi=fig_dpi)
                ax_video.axis("off")

                # Set up progress plot
                ax_progress.set_ylabel("Progress", fontsize=24, fontweight="bold")
                ax_progress.set_xlabel("Timestep", fontsize=24, fontweight="bold")

                title_parts = [f"Task: {task}, Quality: {quality_label}"]
                if is_roboarena and partial_success is not None:
                    title_parts.append(f"Partial Success: {partial_success:.2f}")
                title_parts.append(f"MSE: {traj_mse:.3f}, r: {traj_pearson:.3f}, sp: {traj_spearman:.3f}")
                ax_progress.set_title("\n".join(title_parts), fontsize=20, fontweight="bold", pad=30)
                ax_progress.set_ylim(0, 1)
                ax_progress.set_xlim(0, max(len(last_preds), 1))
                ax_progress.set_yticks([0, 1])
                ax_progress.spines["right"].set_visible(False)
                ax_progress.spines["top"].set_visible(False)
                ax_progress.tick_params(axis="both", which="major", labelsize=18)

                im_video = ax_video.imshow(frames_rgb[0])
                (line_progress,) = ax_progress.plot([], [], linewidth=4, color="blue")

                line_success = None
                if has_success_data:
                    last_success_probs = np.array(success_probs)
                    last_success_labels = np.array(success_labels) if success_labels is not None else None

                    ax_success.set_ylabel("Success Probability", fontsize=24, fontweight="bold")
                    ax_success.set_xlabel("Timestep", fontsize=24, fontweight="bold")
                    ax_success.set_title("Success Prediction", fontsize=20, fontweight="bold", pad=30)
                    ax_success.set_ylim(0, 1)
                    ax_success.set_xlim(0, max(len(last_success_probs), 1))
                    ax_success.set_yticks([0, 1])
                    ax_success.spines["right"].set_visible(False)
                    ax_success.spines["top"].set_visible(False)
                    ax_success.tick_params(axis="both", which="major", labelsize=18)

                    (line_success,) = ax_success.plot([], [], linewidth=4, color="green", label="Predicted")
                    if last_success_labels is not None:
                        ax_success.plot(
                            range(len(last_success_labels)),
                            last_success_labels,
                            linewidth=2,
                            color="red",
                            linestyle="--",
                            label="Ground Truth",
                            alpha=0.7,
                        )
                    ax_success.legend(fontsize=16)

                def animate(frame_idx):
                    # Update video frame
                    im_video.set_array(frames_rgb[frame_idx])

                    # Update progress plot up to current frame
                    max_idx = min(frame_idx + 1, len(last_preds))
                    line_progress.set_data(range(max_idx), last_preds[:max_idx])

                    # Update success plot up to current frame if available
                    if has_success_data and line_success is not None:
                        max_idx_success = min(frame_idx + 1, len(last_success_probs))
                        line_success.set_data(range(max_idx_success), last_success_probs[:max_idx_success])
                        return [im_video, line_progress, line_success]

                    return [im_video, line_progress]

                # Create animation
                anim = animation.FuncAnimation(
                    fig_anim, animate, frames=len(frames_rgb), interval=500, blit=True, repeat=True
                )

                Writer = animation.writers["ffmpeg"]
                writer = Writer(fps=2, metadata=dict(artist="RFM"), bitrate=5000)
                anim.save(video_path, writer=writer, dpi=fig_dpi)
                plt.close(fig_anim)

                saved_count += 1
                logger.debug(f"Saved reward_alignment video: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to save video {video_path}: {e}")

        if saved_count > 0:
            logger.info(f"Saved {saved_count} reward_alignment videos to {output_dir}/reward_alignment_videos/")

    def _save_policy_ranking_incorrect_pairs(self, task_groups, eval_results, output_dir, ds_name, is_roboarena):
        """Save incorrectly and correctly ranked policy pairs to disk.

        Finds pairs where the predicted reward ordering doesn't match the ground truth ordering:
        - For non-RoboArena: successful < failure, successful < suboptimal, suboptimal < failure
        - For RoboArena: partial_success ordering doesn't match predicted reward ordering

        Saves final frames side by side with metadata.
        Saves up to 10 incorrect pairs and 10 correct pairs (randomly sampled).
        """
        quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}
        incorrect_pairs = []
        correct_pairs = []

        for task, trajectories in task_groups.items():
            if len(trajectories) < 2:
                continue

            if is_roboarena:
                # RoboArena: Check pairs where partial_success ordering doesn't match predicted reward ordering
                for i in range(len(trajectories)):
                    for j in range(i + 1, len(trajectories)):
                        traj1 = trajectories[i]
                        traj2 = trajectories[j]

                        partial1 = traj1.get("partial_success")
                        partial2 = traj2.get("partial_success")
                        pred1 = traj1.get("final_predicted_reward")
                        pred2 = traj2.get("final_predicted_reward")

                        if partial1 is None or partial2 is None or pred1 is None or pred2 is None:
                            continue

                        # Skip if partial_success values are the same
                        if partial1 == partial2:
                            continue

                        # Check if ordering is incorrect
                        partial_order_correct = partial1 > partial2  # traj1 should have higher partial_success
                        pred_order = pred1 > pred2  # traj1 has higher predicted reward

                        # Check if ranking is correct or incorrect
                        if partial_order_correct != pred_order:
                            incorrect_pairs.append({
                                "task": task,
                                "traj1": traj1,
                                "traj2": traj2,
                                "partial1": partial1,
                                "partial2": partial2,
                                "pred1": pred1,
                                "pred2": pred2,
                                "error_type": "partial_success_mismatch",
                            })
                        else:
                            correct_pairs.append({
                                "task": task,
                                "traj1": traj1,
                                "traj2": traj2,
                                "partial1": partial1,
                                "partial2": partial2,
                                "pred1": pred1,
                                "pred2": pred2,
                                "error_type": "partial_success_correct",
                            })
            else:
                # Non-RoboArena: Check pairs where quality ordering doesn't match predicted reward ordering
                for i in range(len(trajectories)):
                    for j in range(i + 1, len(trajectories)):
                        traj1 = trajectories[i]
                        traj2 = trajectories[j]

                        quality1 = traj1.get("quality_label")
                        quality2 = traj2.get("quality_label")
                        pred1 = traj1.get("final_reward")
                        pred2 = traj2.get("final_reward")

                        if quality1 is None or quality2 is None or pred1 is None or pred2 is None:
                            continue

                        order1 = quality_order.get(quality1, 0)
                        order2 = quality_order.get(quality2, 0)

                        # Skip if same quality
                        if order1 == order2:
                            continue

                        # Check if ordering is incorrect
                        quality_order_correct = order1 > order2  # traj1 should have higher quality
                        pred_order = pred1 > pred2  # traj1 has higher predicted reward

                        # Check if ranking is correct or incorrect
                        error_type = f"{quality1}_vs_{quality2}"
                        if quality_order_correct != pred_order:
                            incorrect_pairs.append({
                                "task": task,
                                "traj1": traj1,
                                "traj2": traj2,
                                "quality1": quality1,
                                "quality2": quality2,
                                "pred1": pred1,
                                "pred2": pred2,
                                "error_type": error_type,
                            })
                        else:
                            correct_pairs.append({
                                "task": task,
                                "traj1": traj1,
                                "traj2": traj2,
                                "quality1": quality1,
                                "quality2": quality2,
                                "pred1": pred1,
                                "pred2": pred2,
                                "error_type": f"{error_type}_correct",
                            })

        max_pairs = 10

        def sample_diverse_pairs(pairs, max_count):
            """Sample pairs ensuring each trajectory appears at most once."""
            if len(pairs) <= max_count:
                return pairs

            # Track which trajectory IDs we've already used
            used_traj_ids = set()
            selected_pairs = []

            # Shuffle pairs to randomize selection
            shuffled_pairs = pairs.copy()
            random.shuffle(shuffled_pairs)

            for pair in shuffled_pairs:
                if len(selected_pairs) >= max_count:
                    break

                traj1_id = pair["traj1"].get("id")
                traj2_id = pair["traj2"].get("id")

                # Check if either trajectory has been used
                if traj1_id not in used_traj_ids and traj2_id not in used_traj_ids:
                    selected_pairs.append(pair)
                    used_traj_ids.add(traj1_id)
                    used_traj_ids.add(traj2_id)

            # If we haven't filled up to max_count, add remaining pairs even if they repeat
            if len(selected_pairs) < max_count:
                remaining = [p for p in shuffled_pairs if p not in selected_pairs]
                needed = max_count - len(selected_pairs)
                selected_pairs.extend(remaining[:needed])

            return selected_pairs

        incorrect_pairs = sample_diverse_pairs(incorrect_pairs, max_pairs)
        correct_pairs = sample_diverse_pairs(correct_pairs, max_pairs)

        if not incorrect_pairs and not correct_pairs:
            logger.info(f"No pairs found for policy_ranking/{ds_name}")
            return

        # Create output directories
        save_dir = os.path.join(output_dir, "policy_ranking_viz", ds_name)
        os.makedirs(save_dir, exist_ok=True)

        def save_pair_visualization(pair, idx, pair_type, save_dir):
            """Helper function to save a single pair visualization.

            Args:
                pair: Dictionary containing traj1, traj2, task, and metadata
                idx: Index of the pair
                pair_type: "incorrect" or "correct"
                save_dir: Directory to save the visualization

            Returns:
                True if saved successfully, False otherwise
            """
            traj1 = pair["traj1"]
            traj2 = pair["traj2"]
            task = pair["task"]

            # Load final frames from video paths
            video_path1 = traj1.get("video_path")
            video_path2 = traj2.get("video_path")

            if not video_path1 or not video_path2:
                return False

            try:
                # Load frames and get final frame
                frames1 = load_frames_from_npz(video_path1)
                frames2 = load_frames_from_npz(video_path2)

                # Get final frame (last frame in sequence)
                final_frame1 = frames1[-1] if len(frames1.shape) == 4 else frames1
                final_frame2 = frames2[-1] if len(frames2.shape) == 4 else frames2

                # Ensure frames are in (H, W, C) format
                if len(final_frame1.shape) == 3 and final_frame1.shape[0] == 3:
                    final_frame1 = final_frame1.transpose(1, 2, 0)
                if len(final_frame2.shape) == 3 and final_frame2.shape[0] == 3:
                    final_frame2 = final_frame2.transpose(1, 2, 0)

                # Ensure uint8 and correct range
                if final_frame1.dtype != np.uint8:
                    if final_frame1.max() <= 1.0:
                        final_frame1 = (final_frame1 * 255).astype(np.uint8)
                    else:
                        final_frame1 = np.clip(final_frame1, 0, 255).astype(np.uint8)
                if final_frame2.dtype != np.uint8:
                    if final_frame2.max() <= 1.0:
                        final_frame2 = (final_frame2 * 255).astype(np.uint8)
                    else:
                        final_frame2 = np.clip(final_frame2, 0, 255).astype(np.uint8)

                # Create matplotlib figure with two subplots side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                # Display frames
                ax1.imshow(final_frame1)
                ax1.axis("off")
                ax2.imshow(final_frame2)
                ax2.axis("off")

                # Build labels for each trajectory
                traj_id1 = traj1.get("id", "unknown")
                traj_id2 = traj2.get("id", "unknown")

                if is_roboarena:
                    label1_parts = [
                        f"Task: {task}",
                        f"Partial Success: {pair['partial1']:.2f}",
                        f"Predicted Reward: {pair['pred1']:.3f}",
                        f"ID: {traj_id1}",
                    ]
                    label2_parts = [
                        f"Task: {task}",
                        f"Partial Success: {pair['partial2']:.2f}",
                        f"Predicted Reward: {pair['pred2']:.3f}",
                        f"ID: {traj_id2}",
                    ]
                else:
                    label1_parts = [
                        f"Task: {task}",
                        f"Quality: {pair['quality1']}",
                        f"Predicted Reward: {pair['pred1']:.3f}",
                        f"ID: {traj_id1}",
                    ]
                    label2_parts = [
                        f"Task: {task}",
                        f"Quality: {pair['quality2']}",
                        f"Predicted Reward: {pair['pred2']:.3f}",
                        f"ID: {traj_id2}",
                    ]

                # Add labels above frames
                ax1.set_title("\n".join(label1_parts), fontsize=14, fontweight="bold", pad=10)
                ax2.set_title("\n".join(label2_parts), fontsize=14, fontweight="bold", pad=10)

                # Add title based on pair type
                error_type = pair["error_type"]
                title_prefix = "Incorrectly" if pair_type == "incorrect" else "Correctly"
                fig.suptitle(f"{title_prefix} Ranked Pair: {error_type}", fontsize=16, fontweight="bold", y=0.98)

                plt.tight_layout()

                # Save figure
                filename = f"{task}_{error_type}_{idx}.png"
                # Sanitize filename
                filename = filename.replace("/", "_").replace("\\", "_")
                save_path = os.path.join(save_dir, filename)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                return True
            except Exception as e:
                logger.warning(f"Failed to save {pair_type} pair {idx} for task {task}: {e}")
                return False

        # Save incorrect pairs
        saved_incorrect = 0
        for idx, pair in enumerate(incorrect_pairs):
            if save_pair_visualization(pair, idx, "incorrect", save_dir):
                saved_incorrect += 1

        # Save correct pairs
        saved_correct = 0
        for idx, pair in enumerate(correct_pairs):
            if save_pair_visualization(pair, idx, "correct", save_dir):
                saved_correct += 1

        if saved_incorrect > 0 or saved_correct > 0:
            logger.info(
                f"Saved {saved_incorrect} incorrectly ranked pairs and {saved_correct} correctly ranked pairs to {save_dir}/"
            )

    def _compute_and_log_eval_metrics(self, eval_type, eval_results, ds_name, eval_step, output_dir=None):
        """Compute metrics and create visualizations for evaluation results."""
        # Initialize variables to None to ensure they exist for cleanup
        plots = None
        video_frames_list = None
        trajectory_progress_data = None
        task_groups = None
        task_details = None
        confusion_plot = None
        confusion_matrix = None

        if eval_type == "reward_alignment":
            eval_metrics, plots, video_frames_list, trajectory_progress_data = compute_eval_metrics(
                eval_type, eval_results, self.config.data.progress_pred_type
            )
            # log_memory_usage(f"After compute_eval_metrics (reward_alignment)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            # Save videos to disk if output_dir is provided
            if output_dir is not None and video_frames_list:
                self._save_reward_alignment_videos(
                    video_frames_list, plots, eval_results, output_dir, ds_name, trajectory_progress_data
                )

            # Build rows of (video, figure)
            rows = []
            for plot, frames in zip(plots, video_frames_list):
                if frames is not None:
                    rows.append((frames, plot))

            if rows and self.logger.enabled("wandb"):
                self.logger.log_video_table(
                    f"reward_alignment_samples/{ds_name}",
                    videos_and_figures=rows,
                    columns=["video", "progress_plot"],
                    step=eval_step,
                )

            # Create and log 3x3 grid of videos with progress overlays
            if video_frames_list and self.logger.enabled("wandb"):
                grid_video = create_video_grid_with_progress(
                    video_frames_list,
                    trajectory_progress_data=trajectory_progress_data,
                    grid_size=(3, 3),
                    max_videos=9,
                    progress_key_pred="progress_pred",
                    progress_key_target="target_progress",
                )
                if grid_video is not None:
                    self.logger.log_video(
                        f"reward_alignment_grid/{ds_name}",
                        grid_video,
                        fps=2,
                        step=eval_step,
                    )
                    del grid_video

            # For tensorboard (no table support), log each video and its figure separately
            if self.logger.enabled("tensorboard"):
                for idx, frames in enumerate(video_frames_list):
                    if frames is not None:
                        self.logger.log_video(
                            f"reward_alignment_video/{ds_name}/{idx}",
                            frames,
                            fps=2,
                            step=eval_step,
                        )
                for idx, plot in enumerate(plots):
                    self.logger.log_figure(f"{ds_name}/reward_alignment_plot/{idx}", plot, step=eval_step)
            # Close all plots to avoid accumulating open figures
            for plot in plots:
                plt.close(plot)

            # Explicitly delete to free memory and set to None for outer cleanup
            # log_memory_usage(f"Before deleting plots/videos")
            del plots, video_frames_list, trajectory_progress_data, rows
            plots = None
            video_frames_list = None
            trajectory_progress_data = None
            # log_memory_usage(f"After deleting plots/videos")
        elif eval_type == "policy_ranking":
            # create task groups from eval_results
            eval_metrics, task_groups, task_details = compute_eval_metrics(
                eval_type, eval_results, self.config.data.progress_pred_type
            )
            # log_memory_usage(f"After compute_eval_metrics (policy_ranking)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            # Check if this is roboarena by checking if task_groups have partial_reward field
            is_roboarena = False
            if task_groups:
                first_group = next(iter(task_groups.values()))
                if first_group and "partial_success" in first_group[0]:
                    is_roboarena = True

            data = []
            if is_roboarena:
                # RoboArena visualization: show partial vs predicted rewards
                for task, group in task_groups.items():
                    partial_successes = np.array([t["partial_success"] for t in group]).round(2)
                    predicted_rewards = np.array([t["final_predicted_reward"] for t in group]).round(2)
                    partial_successes = partial_successes.tolist()
                    predicted_rewards = predicted_rewards.tolist()
                    data.append([task, f"partial:{partial_successes}", f"predicted:{predicted_rewards}"])
                columns = ["task", "partial_successes", "predicted_rewards"]
            else:
                # Standard policy ranking visualization: show quality labels and rewards
                for task, group in task_groups.items():
                    quality_to_rews = collections.defaultdict(list)
                    for t in group:
                        rew = t["final_reward"]
                        quality_to_rews[t["quality_label"]].append(rew)

                    for q, r in quality_to_rews.items():
                        quality_to_rews[q] = np.array(r).round(2).tolist()
                    quality_to_rews = ",".join([f"{q}:{r}" for q, r in quality_to_rews.items()])

                    # Get task details for differences
                    task_detail = task_details.get(task, {})
                    succ_subopt = task_detail.get("succ_subopt_diff")
                    subopt_fail = task_detail.get("subopt_fail_diff")
                    succ_fail = task_detail.get("succ_fail_diff")

                    # Format differences
                    diff_str = []
                    if succ_subopt is not None:
                        diff_str.append(f"succ-subopt:{succ_subopt:.2f}")
                    if subopt_fail is not None:
                        diff_str.append(f"subopt-fail:{subopt_fail:.2f}")
                    if succ_fail is not None:
                        diff_str.append(f"succ-fail:{succ_fail:.2f}")
                    diff_str = ",".join(diff_str) if diff_str else "N/A"

                    data.append([task, quality_to_rews, diff_str])
                columns = ["task", "quality_and_rews", "avg_differences"]

            table_name = f"policy_ranking_samples/{ds_name}"

            self.logger.log_table(
                table_name,
                data=data,
                columns=columns,
                step=eval_step,
            )

            # Create and log grid of frame pairs with progress annotations
            if self.logger.enabled("wandb"):
                grid_image = create_policy_ranking_grid(eval_results, grid_size=(2, 2), max_samples=4)
                if grid_image is not None:
                    self.logger.log_image(
                        f"policy_ranking_grid/{ds_name}",
                        grid_image,
                        step=eval_step,
                    )
                    del grid_image

            # Save incorrectly ranked pairs to disk if output_dir is provided
            if output_dir is not None:
                self._save_policy_ranking_incorrect_pairs(task_groups, eval_results, output_dir, ds_name, is_roboarena)

            # log_memory_usage(f"Before deleting policy_ranking data")
            del data, task_groups, task_details
            task_groups = None
            task_details = None
            # log_memory_usage(f"After deleting policy_ranking data")
        elif eval_type == "confusion_matrix":
            confusion_plot, confusion_matrix = compute_eval_metrics(
                eval_type, eval_results, self.config.data.progress_pred_type
            )
            eval_metrics = {}  # no eval metrics
            # log_memory_usage(f"After compute_eval_metrics (confusion_matrix)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            self.logger.log_figure(f"eval_cm/{ds_name}", confusion_plot, step=eval_step)
            plt.close(confusion_plot)
            # log_memory_usage(f"Before deleting confusion_matrix data")
            del confusion_plot, confusion_matrix
            confusion_plot = None
            confusion_matrix = None
            # log_memory_usage(f"After deleting confusion_matrix data")
        elif "quality_preference" in eval_type:
            # quality_preference returns metrics, task_groups, and task_details
            eval_metrics, task_groups, task_details = compute_eval_metrics(
                eval_type, eval_results, self.config.data.progress_pred_type
            )
            # log_memory_usage(f"After compute_eval_metrics (quality_preference)")

            banner(
                "Completed evaluation",
                f"{eval_type} evaluation: {len(eval_results)} samples",
                "Metrics",
                f"{eval_metrics}",
                inner_padding=1,
            )

            data = []
            for task, details in task_details.items():
                task_acc = details["preference_accuracy"]
                quality_accs = details["quality_accuracies"]
                quality_accs_str = ",".join([f"{k}:{round(v, 3)}" for k, v in quality_accs.items()])
                num_correct = details["num_correct"]
                num_total = details["num_total"]
                data.append([
                    task,
                    round(task_acc, 3),
                    quality_accs_str if quality_accs_str else "N/A",
                    f"{num_correct}/{num_total}",
                ])
            columns = ["task", "preference_accuracy", "quality_accuracies", "num_correct/total"]

            table_name = f"quality_preference_samples/{ds_name}"

            self.logger.log_table(
                table_name,
                data=data,
                columns=columns,
                step=eval_step,
            )
            # log_memory_usage(f"Before deleting quality_preference data")
            del data, task_groups, task_details
            task_groups = None
            task_details = None
            # log_memory_usage(f"After deleting quality_preference data")
        elif eval_type == "similarity_score":
            # similarity_score returns metrics, task_groups, and task_details
            eval_metrics, task_groups, task_details = compute_eval_metrics(
                eval_type, eval_results, self.config.data.progress_pred_type
            )
            # log_memory_usage(f"After compute_eval_metrics (similarity_score)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            # Create wandb table for similarity score results
            data = []
            for task, group in task_groups.items():
                task_margin = task_details.get(task, {}).get("avg_margin", 0.0)
                task_same_task_score = task_details.get(task, {}).get("avg_same_task_score", 0.0)
                task_diff_task_score = task_details.get(task, {}).get("avg_diff_task_score", 0.0)
                num_pairs = task_details.get(task, {}).get("num_pairs", 0)
                data.append([
                    task,
                    round(task_margin, 3),
                    round(task_same_task_score, 3),
                    round(task_diff_task_score, 3),
                    num_pairs,
                ])
            columns = ["task", "avg_margin", "avg_same_task_score", "avg_diff_task_score", "num_pairs"]
            self.logger.log_table(
                f"similarity_score_samples/{ds_name}",
                data=data,
                columns=columns,
                step=eval_step,
            )
            # log_memory_usage(f"Before deleting similarity_score data")
            del data, task_groups, task_details
            task_groups = None
            task_details = None
            # log_memory_usage(f"After deleting similarity_score data")
        else:
            raise ValueError(f"Unsupported eval type: {eval_type}")

        # Clean up eval-specific outputs
        if plots is not None:
            del plots
        if video_frames_list is not None:
            del video_frames_list
        if trajectory_progress_data is not None:
            del trajectory_progress_data
        if task_groups is not None:
            del task_groups
        if task_details is not None:
            del task_details
        if confusion_plot is not None:
            del confusion_plot
        if confusion_matrix is not None:
            del confusion_matrix

        return eval_metrics

    def _save_eval_results_json(self, eval_results, eval_type, ds_name, output_dir):
        """Save eval_results as JSON file.

        Args:
            eval_results: List of evaluation result dictionaries
            eval_type: Type of evaluation (e.g., "reward_alignment", "policy_ranking")
            ds_name: Dataset name
            output_dir: Directory to save the JSON file
        """

        def serialize_value(value):
            """Recursively serialize a value to JSON-compatible format."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(v) for v in value]
            elif isinstance(value, (int, float, str, type(None))):
                return value
            else:
                # Fallback: try to convert to string
                return str(value)

        # Serialize eval_results
        serialized_results = [serialize_value(result) for result in eval_results]

        # Create output directory if it doesn't exist
        eval_results_dir = os.path.join(output_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)

        # Create filename: {eval_type}_{ds_name}.json
        filename = f"{eval_type}_{ds_name}.json"
        filepath = os.path.join(eval_results_dir, filename)

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(serialized_results, f, indent=2)
        logger.info(f"Saved {len(eval_results)} eval results to: {filepath}")

    def _cleanup_eval_dataset(self, dataset, dataloader, eval_results):
        """Clean up dataset, dataloader, and eval_results after evaluation."""
        logger.info(f"  Cleaning up dataset and eval_results")
        # log_memory_usage(f"Before cleanup")

        # Aggressive cleanup to prevent memory leaks
        # First, delete eval_results which can be large
        del eval_results

        # For the dataloader, we need to ensure worker processes are shut down
        # The accelerator.prepare() wraps the dataloader, so we need to clean both
        # Access the underlying dataloader if it exists and has workers
        try:
            if hasattr(dataloader, "_loader"):
                # Accelerator-wrapped dataloader
                underlying_dl = dataloader._loader
            else:
                underlying_dl = dataloader

            # Shutdown workers if they exist
            if hasattr(underlying_dl, "_iterator") and underlying_dl._iterator is not None:
                underlying_dl._iterator._shutdown_workers()
                underlying_dl._iterator = None
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not explicitly shutdown workers: {e}")

        # Delete dataloader and dataset
        del dataloader, dataset
        # log_memory_usage(f"After deleting dataloader and dataset")

        # Force garbage collection
        import gc

        gc.collect()
        # log_memory_usage(f"After first gc.collect()")
        gc.collect()  # Call twice for cyclic references
        # log_memory_usage(f"After second gc.collect()")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _run_single_eval_dataset(self, eval_type, eval_dataset, eval_step, output_dir=None):
        """Run evaluation for a single dataset."""
        logger.info(f"  Processing dataset: {eval_dataset}")
        # log_memory_usage(f"Before dataset {eval_dataset}")

        # Get dataset name for mapping
        dataset_for_mapping = eval_dataset[0] if isinstance(eval_dataset, list) else eval_dataset
        ds_name = DS_SHORT_NAME_MAPPING.get(dataset_for_mapping, dataset_for_mapping)
        timing_key = f"time/eval_dataset/{eval_type}/{ds_name}"

        with _timer(timing_key, timing_raw=self.timing_raw):
            # Setup dataset and dataloader
            dataset, dataloader = self._setup_eval_dataset(eval_type, eval_dataset)

            eval_results = []
            batch_idx = 0
            # Create tqdm iterator explicitly so we can close it properly
            dataloader_iter = tqdm(
                dataloader,
                desc=f"Running {eval_type}, ds: {eval_dataset}, batch size: {self.config.training.per_device_eval_batch_size}",
                disable=not is_rank_0(),
            )

            for batch in dataloader_iter:
                logger.trace(f"  Processing batch {batch_idx}")
                # if batch_idx % 10 == 0:  # Log memory every 10 batches
                #     log_memory_usage(f"Batch {batch_idx}/{len(dataloader)}")

                batch = self._prepare_inputs(batch)

                # Log batch composition
                num_pref = batch.get("num_preferences", 0)
                num_prog = batch.get("num_progress", 0)
                num_sim = batch.get("num_similarities", 0)
                logger.trace(f"  Batch {batch_idx}: pref={num_pref}, prog={num_prog}, sim={num_sim}")
                batch_idx += 1

                # Process batch based on eval type
                if eval_type in ["reward_alignment", "policy_ranking", "confusion_matrix"]:
                    batch_results, outputs = self._process_batch_progress_eval(batch, eval_type)
                    eval_results.extend(batch_results)
                elif "quality_preference" in eval_type:
                    batch_results, outputs = self._process_batch_preference_eval(batch)
                    eval_results.extend(batch_results)
                elif eval_type == "similarity_score":
                    batch_results, outputs = self._process_batch_similarity_eval(batch)
                    eval_results.extend(batch_results)

                # Clean up batch tensors and free memory after each batch
                # This is critical for VQA with generation to prevent OOM
                del batch, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # # Log memory after cleanup every 10 batches
                # if (batch_idx - 1) % 10 == 0:
                #     log_memory_usage(f"After batch {batch_idx - 1} cleanup")

            # Close tqdm iterator to release any held references
            dataloader_iter.close()
            del dataloader_iter

            logger.info(f"  Finished processing {len(eval_results)} eval results")
            # log_memory_usage(f"After eval loop, before compute_eval_metrics")

            # Compute metrics and create visualizations (only on main process)
            eval_metrics = {}
            if self.accelerator.is_main_process:
                # Use output_dir parameter if provided, otherwise fall back to config
                actual_output_dir = output_dir if output_dir is not None else getattr(self.config, "output_dir", None)
                eval_metrics = self._compute_and_log_eval_metrics(
                    eval_type, eval_results, ds_name, eval_step, output_dir=actual_output_dir
                )

                # Save eval_results as JSON if output_dir is available
                if actual_output_dir is not None:
                    self._save_eval_results_json(eval_results, eval_type, ds_name, actual_output_dir)

            # Cleanup
            self._cleanup_eval_dataset(dataset, dataloader, eval_results)

            # log_memory_usage(f"After cleanup for {eval_dataset}")

            # Store timing for this eval_dataset
            eval_dataset_time = self.timing_raw.get(timing_key, 0.0)
            logger.info(f"  Finished {eval_type} for {eval_dataset} (took {eval_dataset_time:.2f} seconds)")
            logger.info("-" * 80)

            return eval_metrics, ds_name

    def _run_custom_evaluations(self, eval_step=None, output_dir=None):
        """
        Run custom evaluations.

        Args:
            eval_step: Step number to use for logging. If None, uses self.state.global_step.
                      This ensures consistent step logging to prevent wandb warnings.
            output_dir: Optional directory to save evaluation outputs (e.g., videos for reward_alignment).
        """
        if eval_step is None:
            eval_step = self.state.global_step

        logger.info("=" * 100)
        logger.info("STARTING CUSTOM EVALUATIONS")
        # log_memory_usage("Before custom evaluations")
        logger.info("=" * 100)

        metrics = collections.defaultdict(dict)
        eval_types = self.config.custom_eval.eval_types

        EVAL_TYPE_SHORT = {
            "reward_alignment": "rew_align",
            "confusion_matrix": "cm",
            "policy_ranking": "p_rank",
            "quality_preference": "pref",
            "quality_preference_roboarena": "pref_robo",
            "similarity_score": "sim_score",
        }

        banner("Running custom evaluations", f"Custom evaluations: {eval_types}")

        eval_type_timings = {}
        eval_dataset_timings = {}

        for eval_type in eval_types:
            logger.info("=" * 80)
            logger.info(f"Running evaluation for: {eval_type}")
            # log_memory_usage(f"Before {eval_type}")
            logger.info("=" * 80)

            datasets = getattr(self.config.custom_eval, eval_type)
            eval_datasets_name = resolve_dataset_keys(datasets, split="eval")

            with _timer(f"time/eval_type/{eval_type}", timing_raw=self.timing_raw):
                for eval_dataset in eval_datasets_name:
                    eval_metrics, ds_name = self._run_single_eval_dataset(
                        eval_type, eval_dataset, eval_step, output_dir=output_dir
                    )
                    metrics[ds_name][eval_type] = eval_metrics

                    # Store timing for this eval_dataset
                    dataset_for_mapping = eval_dataset[0] if isinstance(eval_dataset, list) else eval_dataset
                    ds_name_mapped = DS_SHORT_NAME_MAPPING.get(dataset_for_mapping, dataset_for_mapping)
                    timing_key = f"time/eval_dataset/{eval_type}/{ds_name_mapped}"
                    eval_dataset_time = self.timing_raw.get(timing_key, 0.0)
                    eval_dataset_timings[timing_key] = eval_dataset_time

                # log_memory_usage(f"After completing all datasets for {eval_type}")

                # Store timing for this eval_type
                eval_type_time = self.timing_raw.get(f"time/eval_type/{eval_type}", 0.0)
                eval_type_timings[f"time/eval_type/{eval_type}"] = eval_type_time
                logger.info(f"Finished eval_type: {eval_type} (took {eval_type_time:.2f} seconds)")
                logger.info("=" * 80)

        flat_metrics = {}
        for ds_name, eval_type_metric in metrics.items():
            for eval_type, metric in eval_type_metric.items():
                eval_type_short = EVAL_TYPE_SHORT[eval_type]
                # Add to flat metrics dict with full names
                for k, v in metric.items():
                    if isinstance(v, (int, float)):
                        metric_name = f"eval_{eval_type_short}/{k}_{ds_name}"
                        flat_metrics[metric_name] = v

        # Prepare metrics for callbacks (all processes should have the same metrics)
        callback_metrics = flat_metrics

        # Prepare wandb metrics and log (only on main process)
        if self.accelerator.is_main_process:
            # Convert callback_metrics to float for wandb logging
            to_log = {k: float(v) for k, v in callback_metrics.items()}
            to_log["epoch"] = self.state.epoch

            # Add timing metrics
            for timing_key, timing_value in eval_type_timings.items():
                to_log[timing_key] = float(timing_value)
            for timing_key, timing_value in eval_dataset_timings.items():
                to_log[timing_key] = float(timing_value)

            self.logger.log_scalars(to_log, step=eval_step)

            # Log timing summary to console
            if is_rank_0():
                logger.info("=" * 80)
                logger.info("Custom Evaluation Timing Summary")
                logger.info("=" * 80)
                logger.info("Per eval_type:")
                for timing_key, timing_value in sorted(eval_type_timings.items()):
                    logger.info(f"  {timing_key}: {timing_value:.2f} seconds")
                logger.info("Per eval_dataset:")
                for timing_key, timing_value in sorted(eval_dataset_timings.items()):
                    logger.info(f"  {timing_key}: {timing_value:.2f} seconds")
                logger.info("=" * 80)

        banner("Finished running custom evaluations!")
        # log_memory_usage("After all evaluations, before cleanup")

        # Reset model to training mode and clear any cached states to prevent leakage
        self.model.train()
        # Ensure gradients are cleared before returning to training
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Aggressive cleanup to prevent OOM after evaluation
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Call twice
        gc.collect()
        gc.collect()

        # Clean up large objects
        del metrics

        # log_memory_usage("After final cleanup")
        logger.info("=" * 100)
        logger.info("FINISHED CUSTOM EVALUATIONS")
        logger.info("=" * 100)

        # Final synchronization barrier to ensure all processes finish together
        if dist.is_initialized():
            dist.barrier()

        return callback_metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
        eval_step = self.state.global_step

        # Save current training mode and set to eval mode
        was_training = self.model.training
        self.model.eval()
        metrics = {}

        # Run evaluation
        if self.config.training.run_default_eval:
            # Get the evaluation dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            outputs = []
            with _timer("time/evaluate", timing_raw=self.timing_raw):
                with torch.no_grad():
                    for _step, inputs in tqdm(
                        enumerate(eval_dataloader),
                        total=len(eval_dataloader),
                        desc="Evaluating",
                        disable=not is_rank_0(),
                    ):
                        # Move inputs to device
                        inputs = self._prepare_inputs(inputs)

                        _, loss_dicts = self.compute_loss(self.model, inputs, return_outputs=True, training=False)
                        outputs.append(loss_dicts)

            # assume that we already called .item() on the outputs
            keys = list(outputs[0].keys())
            for key in keys:
                metrics[key] = [output[key] for output in outputs if key in output]
                metrics[key] = np.array(metrics[key]).mean()

            # Aggregate metrics across all processes using accelerator
            metrics = reduce_metrics_with_accelerate(metrics, self.accelerator, aggregate_method="mean")
            metrics["time/evaluate"] = self.timing_raw["time/evaluate"]

        # Run the custom evaluations
        custom_eval_should_run = (
            self.config.training.custom_eval_steps
            and self.state.global_step % self.config.training.custom_eval_steps == 0
        )
        if custom_eval_should_run:
            with _timer("time/custom_evaluations", timing_raw=self.timing_raw):
                # Get output_dir from config if available (for offline eval)
                output_dir = getattr(self.config, "output_dir", None)
                custom_metrics = self._run_custom_evaluations(eval_step=eval_step, output_dir=output_dir)

            metrics.update(custom_metrics)
            # Add custom evaluation time
            metrics["time/custom_evaluations"] = self.timing_raw["time/custom_evaluations"]

            if is_rank_0():
                logger.info(f"Custom evaluations took {self.timing_raw['time/custom_evaluations']:.2f} seconds")

        if metrics:
            if is_rank_0():
                banner("Evaluation Results (Aggregated)", inner_padding=1)
                for key, value in metrics.items():
                    logger.info(f"{key}: {value:.6f}")
                logger.info("=" * 50)

            if is_rank_0():
                self.logger.log_scalars(metrics, step=eval_step)

            # Trigger the callback handler with all metrics
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        # CRITICAL: Final barrier OUTSIDE the if metrics block to ensure ALL ranks wait
        # This is the absolute final barrier before returning from evaluate(), ensuring no training can start
        # until all evaluation is completely done, regardless of whether metrics were computed
        if dist.is_initialized():
            logger.debug(f"[Rank {get_rank()}] Waiting at final barrier before returning from evaluate()")
            dist.barrier()
            logger.debug(f"[Rank {get_rank()}] Passed final barrier, about to return from evaluate()")

        # Restore original training mode to prevent state leakage
        self.model.train(was_training)
        # Ensure gradients are cleared before returning to training
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        # Clear any cached states that might persist from evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """Compute loss for separate preference and similarity batches."""
        logger.trace("compute_loss: Starting")

        # Set static graph for DDP on first training step to handle multiple forward passes
        # This is necessary because similarity loss does 2 forward passes (ref_sim and ref_diff)
        if (
            training
            and not self._ddp_static_graph_set
            and getattr(self.accelerator.gradient_state, "sync_gradients", True)
            and hasattr(model, "module")
        ):
            if hasattr(model.module, "_set_static_graph"):
                logger.info("Setting DDP static graph mode for multiple forward passes")
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, "_set_static_graph"):
                logger.info("Setting DDP static graph mode for multiple forward passes")
                model._set_static_graph()
                self._ddp_static_graph_set = True

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})

        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        num_progress = inputs.get("num_progress", 0)

        total_loss = 0
        log_metadata = {}

        logger.trace(
            f"Num preferences: {num_preferences}, Num similarities: {num_similarities}, Num progress: {num_progress}"
        )

        # Compute preference loss if we have preference samples
        if num_preferences > 0 and preference_inputs and self.config.model.train_preference_head:
            with _timer("time/compute_preference_loss", timing_raw=self.timing_raw):
                preference_loss, loss_dict = self._compute_preference_loss(
                    model, preference_inputs, return_outputs=True, training=training
                )
                total_loss += preference_loss
                log_metadata.update(loss_dict)

        # Compute progress loss if we have progress samples
        if num_progress > 0 and progress_inputs and self.config.model.train_progress_head:
            with _timer("time/compute_progress_loss", timing_raw=self.timing_raw):
                progress_loss, loss_dict = self._compute_progress_loss(
                    model, progress_inputs, return_outputs=True, training=training
                )
                total_loss += progress_loss
            log_metadata.update(loss_dict)

        # Compute similarity loss if we have similarity samples
        if num_similarities > 0 and similarity_inputs and self.config.model.train_similarity_head:
            with _timer("time/compute_similarity_loss", timing_raw=self.timing_raw):
                similarity_loss, loss_dict = self._compute_similarity_loss(
                    model, similarity_inputs, return_outputs=True, training=training
                )
                total_loss += similarity_loss
            log_metadata.update(loss_dict)

        for key, value in log_metadata.items():
            logger.trace(f"{key}: {value}, type: {type(value)}")
            if isinstance(value, torch.Tensor):
                logger.trace(f"\t{key}: shape={value.shape}")

        # Check for NaN in total loss before returning
        if torch.isnan(total_loss).any():
            logger.warning(f"NaN detected in total_loss, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {**log_metadata, "total_loss": total_loss.item()}
            return total_loss, extra_info

        return total_loss

    def _compute_success_loss_helper(
        self, success_logits, target_progress, success_labels, progress_loss_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper function to compute success prediction loss.

        Computes binary cross-entropy loss for frames with:
        - progress < min_success (label=0, failure)
        - progress > max_success (label=1, success)
        - ignores frames in between

        Args:
            success_logits: Success prediction logits (can be tensor or list of tensors)
            target_progress: Target progress tensors (can be tensor or list of tensors)
            success_labels: Success labels from batch (computed in collator) (can be tensor or list of tensors)
            progress_loss_mask: Per-sample mask tensor of shape (batch_size,) with 1.0 for samples
                where we should compute progress/success loss (e.g., successful, rewound, different_task)
            aggregate: Whether to return the mean of the losses and accuracies

        Returns:
            tuple: (success_loss, success_accuracy, success_auprc, metrics)
        """
        # Get base thresholds from config
        min_success = self.config.data.min_success

        # Handle Qwen/Molmo downsampling: take every 2nd frame if using Qwen/Molmo and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        # Ensure success_logits matches target_progress length after downsampling
        if ("Qwen" in self.config.model.base_model_id or "Molmo" in self.config.model.base_model_id) and not self.config.data.use_multi_image:
            success_logits = success_logits[:, ::2]
            target_progress = target_progress[:, ::2]
            success_labels = success_labels[:, ::2]

        combined_mask = ((target_progress < min_success) | (success_labels > 0.5)).float()

        if progress_loss_mask is not None:
            combined_mask = combined_mask * progress_loss_mask

        # Clamp logits to prevent extreme values and gradient issues
        success_logits = torch.clamp(success_logits, min=-50.0, max=50.0)

        positive_weight_value = float(getattr(self.config.loss, "success_positive_weight", 1.0))
        pos_weight_tensor = torch.tensor(
            positive_weight_value, device=success_logits.device, dtype=success_logits.dtype
        )

        loss = F.binary_cross_entropy_with_logits(
            success_logits,
            success_labels,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        masked_loss = loss * combined_mask

        # Compute accuracy per sample
        success_preds = (torch.sigmoid(success_logits) > 0.5).float()
        correct = (success_preds == success_labels).float()
        masked_correct = correct * combined_mask

        # Compute AUPRC (Area Under Precision-Recall Curve)
        # Flatten tensors for AUPRC computation
        success_probs = torch.sigmoid(success_logits)
        success_probs_flat = success_probs[combined_mask > 0]
        success_labels_flat = success_labels[combined_mask > 0]

        success_loss = masked_loss.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
        success_acc = masked_correct.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
        success_loss = success_loss.mean()
        success_acc = success_acc.mean()

        # Compute AUPRC across all valid frames
        if success_probs_flat.numel() > 0 and len(torch.unique(success_labels_flat)) > 1:
            auprc = average_precision_score(
                t2n(success_labels_flat),
                t2n(success_probs_flat),
            )
            batch_auprc = torch.tensor(auprc, device=success_loss.device, dtype=torch.float32)
        else:
            batch_auprc = torch.tensor(0.0, device=success_loss.device, dtype=torch.float32)

        metrics = {
            "masked_loss": masked_loss,
            "masked_correct": masked_correct,
        }

        return success_loss, success_acc, batch_auprc, metrics

    def _compute_progress_loss_helper(
        self,
        progress_pred: torch.Tensor,
        target_progress: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper function to compute progress loss.

        Args:
            progress_pred: Progress prediction tensors (can be tensor or list of tensors) of shape (batch_size, seq_len)
            target_progress: Target progress tensors (can be tensor or list of tensors) of shape (batch_size, seq_len)
            mask: Per-sample mask tensor of shape (batch_size,) with 1.0 for samples where we should compute loss

        Returns:
            tuple: (masked_loss, spearman_correlations, metrics)
        """
        # Handle Qwen downsampling: take every 2nd frame if using Qwen and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        if ("Qwen" in self.config.model.base_model_id or "Molmo" in self.config.model.base_model_id) and not self.config.data.use_multi_image:
            target_progress = target_progress[:, ::2]
            mask = mask[:, ::2]

        # If predict_last_frame_progress is True, only compute loss for the last frame
        last_frame_mask = None
        if self.config.loss.predict_last_frame_progress:
            # Create a mask that only selects the last frame for each sequence
            last_frame_mask = torch.zeros_like(target_progress, dtype=torch.float32)
            last_frame_mask[:, -1] = 1.0  # Set last frame to 1.0 for all sequences
            mask = mask * last_frame_mask

        # Compute MSE loss per frame
        loss_per_frame = F.mse_loss(progress_pred.float(), target_progress.float(), reduction="none")
        masked_loss = loss_per_frame * mask

        if mask.shape[1] != target_progress.shape[1]:
            repeated_mask = mask.repeat(1, target_progress.shape[1])
        else:
            repeated_mask = mask
        masked_spearman_corr = compute_spearman_correlation(
            progress_pred, target_progress, aggregate=False, mask=repeated_mask
        )
        masked_spearman_corr = masked_spearman_corr.detach()

        # Average per sample, then take mean across batch
        # TODO: might need to change this if the mask is per timestep too
        progress_loss = masked_loss.mean(dim=1).sum(dim=0) / mask.sum()
        spearman_corr = masked_spearman_corr.mean()

        # Keep track of the per-sample metrics
        metrics = {"masked_loss": masked_loss, "masked_spearman_corr": masked_spearman_corr}

        return progress_loss, spearman_corr, metrics

    def _add_stratified_metrics(
        self,
        outputs_dict: Dict,
        prefix: str,
        strategy_values: list | None,
        data_source_values: list,
        metrics: Dict[str, torch.Tensor],
    ) -> None:
        """
        Add stratified metrics (by strategy and data source) to outputs_dict.

        Args:
            outputs_dict: Dictionary to update with metrics
            prefix: Prefix for metric keys (e.g., "train" or "eval")
            strategy_values: List of strategy values to split by (e.g., data_gen_strategy), or None to skip
            data_source_values: List of data source values to split by
            metrics: Dictionary of metric tensors, e.g., {"acc": tensor, "loss": tensor, "margin": tensor}
        """
        device = self.accelerator.device

        # Split by strategy
        if strategy_values is not None:
            strats = set(strategy_values)
            for strat in strats:
                mask = torch.tensor(
                    [1 if s == strat else 0 for s in strategy_values], device=device, requires_grad=False
                )
                # Apply mask to each metric and compute mean
                for metric_name, metric_tensor in metrics.items():
                    masked_metric = metric_tensor[mask == 1].detach()
                    mean_value = masked_metric.mean().item() if masked_metric.numel() > 0 else 0.0
                    outputs_dict[f"{prefix}_strat_{metric_name}/{strat}"] = mean_value

        # Split by data source
        data_sources = set(data_source_values)
        for data_source in data_sources:
            mask = torch.tensor(
                [1 if s == data_source else 0 for s in data_source_values], device=device, requires_grad=False
            )
            # Apply mask to each metric and compute mean
            for metric_name, metric_tensor in metrics.items():
                masked_metric = metric_tensor[mask == 1].detach()
                mean_value = masked_metric.mean().item() if masked_metric.numel() > 0 else 0.0
                outputs_dict[f"{prefix}_ds_{metric_name}/{data_source}"] = mean_value

    def forward_model(self, model, inputs, sample_type="progress"):
        """Forward pass for the model."""
        logger.trace(f"forward_model: Starting forward pass for sample_type={sample_type}")

        with _timer("time/forward", timing_raw=self.timing_raw):
            if "rewind" in self.config.model.base_model_id:
                logger.trace("forward_model: Using ReWiND model path")
                model_output, model_timing_raw = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values", None),
                    pixel_values_videos=inputs.get("pixel_values_videos", None),
                    video_embeddings=inputs.get("video_embeddings", None),
                    text_embeddings=inputs.get("text_embeddings", None),
                    sample_type=sample_type,
                    timing_raw=self.timing_raw,
                )
            else:
                logger.trace("forward_model: Using Qwen/Molmo/RFM model path, calling model forward")
                logger.trace(
                    f"forward_model: input_ids shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}"
                )
                model_output, model_timing_raw = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values", None),
                    pixel_values_videos=inputs.get("pixel_values_videos", None),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    video_grid_thw=inputs.get("video_grid_thw", None),
                    second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                    sample_type=sample_type,
                    timing_raw=self.timing_raw,
                )
                logger.trace("forward_model: Model forward pass completed")

            logger.trace("forward_model: Updating timing and returning")
            self.timing_raw.update(model_timing_raw)
            return model_output, model_timing_raw

    def _compute_progress_loss(self, model, inputs, return_outputs=False, training=True, stratify_by_strategy=True):
        """
        Compute progress prediction loss.

        Args:
            model: The model to use for forward pass
            inputs: Input dictionary containing progress data
            return_outputs: Whether to return detailed outputs dict
            training: Whether in training mode
            stratify_by_strategy: Whether to stratify metrics by data_gen_strategy (default: True)
                                 Set to False for single-frame training where strategies aren't used
        """
        model_output, _ = self.forward_model(model, inputs, sample_type="progress")
        progress_logits = model_output.progress_logits
        progress_pred = progress_logits["A"]
        progress_target = inputs["target_progress"]
        progress_target_mask = inputs["target_progress_mask"].unsqueeze(-1)

        progress_loss, spearman_corr, progress_metrics = self._compute_progress_loss_helper(
            progress_pred, progress_target, progress_target_mask
        )

        final_loss = progress_loss
        if self.config.model.train_success_head:
            success_logits = model_output.success_logits
            success_pred = success_logits["A"]
            success_labels = inputs["success_labels"]

            success_loss, success_accuracy, success_auprc, success_metrics = self._compute_success_loss_helper(
                success_pred,
                progress_target,
                success_labels,
                progress_loss_mask=progress_target_mask,
            )
            final_loss = progress_loss + success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            if training:
                import ipdb

                ipdb.set_trace()
            logger.warning(f"NaN detected in progress loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"
            stratified_metrics = {
                "spearman_corr": progress_metrics["masked_spearman_corr"],
                "prog_loss": progress_metrics["masked_loss"],
            }

            strategy_values = inputs.get("data_gen_strategy") if stratify_by_strategy else None
            self._add_stratified_metrics(
                outputs_dict,
                prefix,
                strategy_values,
                inputs["data_source"],
                stratified_metrics,
            )

            outputs_dict.update({
                f"{prefix}/prog_loss": progress_loss.item(),
                f"{prefix}/spearman_corr": spearman_corr.item(),
            })

            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/success_loss": success_loss.item(),
                    f"{prefix}/success_accuracy": success_accuracy.item(),
                    f"{prefix}/success_auprc": success_auprc.item(),
                })

        if not return_outputs:
            return final_loss

        return final_loss, outputs_dict

    def _compute_preference_loss(self, model, inputs, return_outputs=False, training=True):
        """Compute preference prediction loss using Bradley-Terry model."""
        model_outputs, model_timing_raw = self.forward_model(model, inputs, sample_type="preference")
        progress_logits = model_outputs.progress_logits

        # Get preference labels (1 if first trajectory is preferred, 0 if second trajectory is preferred)
        preference_labels = inputs["preference_labels"]

        # Get preference scores from the preference head
        preference_scores = model_outputs.pref_logits.squeeze(-1)  # [batch_size]

        # Clamp logits to prevent extreme values and gradient issues
        preference_scores = torch.clamp(preference_scores, min=-50.0, max=50.0)

        # Binary cross entropy loss for preference prediction
        preference_loss_all = F.binary_cross_entropy_with_logits(
            preference_scores, preference_labels.float(), reduction="none"
        )
        preference_loss = preference_loss_all.mean()

        final_loss = preference_loss

        # =========================================================================================
        # Compute progress and success loss for the first trajectory in the paired samples
        # =========================================================================================
        target_progress_A = inputs["target_progress_A"]
        target_progress_A_mask = inputs["target_progress_A_mask"].unsqueeze(-1)

        if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
            progress_pred_A = progress_logits["A"]

            progress_loss_A, spearman_corr_A, progress_metrics_A = self._compute_progress_loss_helper(
                progress_pred_A,
                target_progress_A,
                target_progress_A_mask,
            )
            final_loss += progress_loss_A

        if self.config.model.train_success_head:
            success_logits = model_outputs.success_logits
            success_logits = success_logits["A"]
            success_labels_A = inputs["success_labels_A"]

            success_loss, success_accuracy, success_auprc, success_metrics_A = self._compute_success_loss_helper(
                success_logits,
                target_progress_A,
                success_labels_A,
                progress_loss_mask=target_progress_A_mask,
            )
            final_loss += success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            logger.warning(f"NaN detected in preference loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"
            rejected_data_gen_strategy = inputs["rejected_data_gen_strategy"]

            if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
                outputs_dict.update({
                    f"{prefix}/pref_prog_loss": progress_loss_A.item(),
                    f"{prefix}/pref_prog_spearman_corr": spearman_corr_A.item(),
                })

                stratified_progress_metrics = {
                    "spearman_corr": progress_metrics_A["masked_spearman_corr"],
                    "prog_loss": progress_metrics_A["masked_loss"],
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["trajectory_A_data_gen_strategy"],
                    inputs["data_source"],
                    stratified_progress_metrics,
                )

            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/pref_success_loss": success_loss.item(),
                    f"{prefix}/pref_success_accuracy": success_accuracy.item(),
                    f"{prefix}/pref_success_auprc": success_auprc.item(),
                })

                stratified_success_metrics = {
                    "success_loss": success_metrics_A["masked_loss"],
                    "success_acc": success_metrics_A["masked_correct"],
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["trajectory_A_data_gen_strategy"],
                    inputs["data_source"],
                    stratified_success_metrics,
                )

            if preference_loss is not None:
                # Compute preference accuracy for training monitoring
                preference_probs = torch.sigmoid(preference_scores)
                preference_predictions = (preference_probs > 0.5).float()
                preference_accuracy = (preference_predictions == preference_labels).float()

                # Prepare metrics for stratification
                stratified_metrics = {
                    "pref_acc": preference_accuracy,
                    "pref_loss": preference_loss_all,
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    rejected_data_gen_strategy,
                    inputs["data_source"],
                    stratified_metrics,
                )

                outputs_dict.update({
                    f"{prefix}/preference_loss": preference_loss.item(),
                    f"{prefix}/preference_accuracy": preference_accuracy.mean().item(),
                })
                return final_loss, outputs_dict

        return final_loss

    def _compute_similarity_loss(self, model, inputs, return_outputs=False, training=True):
        """Compute similarity scoring loss (DPO-style).

        The inputs are already batched by the collator as [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        We do a single forward pass and then extract scores for ref_sim (even indices) and ref_diff (odd indices).
        """

        logger.trace("computing similarity loss")

        # Check batch size consistency across ranks
        batch_size = inputs.get("input_ids", torch.tensor([])).shape[0] if "input_ids" in inputs else 0
        logger.trace(f"similarity batch size: {batch_size}")

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.trace(f"{key}: {value.shape}")
            else:
                logger.trace(f"{key}: {value}")

        # Single forward pass with batched inputs (already batched by collator)
        # Batch structure: [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        logger.trace("About to call forward_model for similarity")
        batched_outputs, _ = self.forward_model(model, inputs, sample_type="similarity")

        logger.trace("finished forward pass for similarity loss")

        # Extract batch size (number of similarity samples)
        # The batched input has 2x the number of samples (ref_sim + ref_diff for each)
        num_samples = len(inputs.get("data_source", []))
        batch_size = num_samples  # Number of similarity samples

        # Split outputs: even indices are ref_sim, odd indices are ref_diff
        # Handle sim_logits
        sim_logits_ref_sim = (
            batched_outputs.sim_logits[::2] if batched_outputs.sim_logits is not None else None
        )  # Even indices
        sim_logits_ref_diff = (
            batched_outputs.sim_logits[1::2] if batched_outputs.sim_logits is not None else None
        )  # Odd indices

        logger.trace(f"sim_logits_ref_sim: {sim_logits_ref_sim}, shape: {sim_logits_ref_sim.shape}")
        logger.trace(f"sim_logits_ref_diff: {sim_logits_ref_diff}, shape: {sim_logits_ref_diff.shape}")

        # Handle progress_logits
        progress_logits_ref_sim = None
        progress_logits_ref_diff = None
        if batched_outputs.progress_logits is not None and batched_outputs.progress_logits.get("A") is not None:
            progress_A = batched_outputs.progress_logits["A"]
            # Split along batch dimension
            progress_logits_ref_sim = {"A": progress_A[::2], "B": None}
            progress_logits_ref_diff = {"A": progress_A[1::2], "B": None}

        # Handle success_logits
        success_logits_ref_sim = None
        success_logits_ref_diff = None
        if batched_outputs.success_logits is not None and batched_outputs.success_logits.get("A") is not None:
            success_A = batched_outputs.success_logits["A"]
            # Split along batch dimension
            success_logits_ref_sim = {"A": success_A[::2], "B": None}
            success_logits_ref_diff = {"A": success_A[1::2], "B": None}

        model_outputs_ref_sim = ModelOutput(
            sim_logits=sim_logits_ref_sim,
            progress_logits=progress_logits_ref_sim,
            success_logits=success_logits_ref_sim,
        )
        model_outputs_ref_diff = ModelOutput(
            sim_logits=sim_logits_ref_diff,
            progress_logits=progress_logits_ref_diff,
            success_logits=success_logits_ref_diff,
        )

        score_ref_sim = model_outputs_ref_sim.sim_logits.squeeze(-1)
        score_ref_diff = model_outputs_ref_diff.sim_logits.squeeze(-1)

        # Clamp logits to prevent extreme values and gradient issues
        score_ref_sim = torch.clamp(score_ref_sim, min=-50.0, max=50.0)
        score_ref_diff = torch.clamp(score_ref_diff, min=-50.0, max=50.0)

        # Compute DPO-style loss: encourage trajectory sim to be more similar to reference than trajectory diff
        # This assumes trajectory sim is the "better" trajectory (more similar to reference)
        # Use softplus for numerical stability: -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        diff_scores = self.config.training.beta * (score_ref_sim - score_ref_diff)
        diff_scores = torch.clamp(diff_scores, min=-50.0, max=50.0)
        similarity_loss_all = F.softplus(-diff_scores)
        similarity_loss = similarity_loss_all.mean()
        similarity_margin = (score_ref_sim - score_ref_diff).detach()
        final_loss = similarity_loss

        # =========================================================================================
        # Compute progress and success loss for randomly selected trajectory in both ref_sim and ref_diff
        # =========================================================================================
        # Randomly select which trajectory (A or B) to predict progress/success for in each comparison
        # 1.0 means use trajectory A (ref), 0.0 means use trajectory B (sim/diff)
        num_samples = len(inputs["data_source"])
        progress_pred_traj_ref_sim = torch.randint(
            0, 2, (num_samples,), device=self.accelerator.device, dtype=torch.float32
        )
        progress_pred_traj_ref_diff = torch.randint(
            0, 2, (num_samples,), device=self.accelerator.device, dtype=torch.float32
        )
        success_pred_traj_ref_sim = torch.randint(
            0, 2, (num_samples,), device=self.accelerator.device, dtype=torch.float32
        )
        success_pred_traj_ref_diff = torch.randint(
            0, 2, (num_samples,), device=self.accelerator.device, dtype=torch.float32
        )

        # Get target progress and masks for all trajectories
        target_progress_ref = inputs["target_progress_ref"]  # [batch_size, seq_len]
        target_progress_sim = inputs["target_progress_sim"]  # [batch_size, seq_len]
        target_progress_diff = inputs["target_progress_diff"]  # [batch_size, seq_len]
        target_progress_ref_mask = inputs["target_progress_ref_mask"].unsqueeze(-1)  # [batch_size, 1]
        target_progress_sim_mask = inputs["target_progress_sim_mask"].unsqueeze(-1)  # [batch_size, 1]
        target_progress_diff_mask = inputs["target_progress_diff_mask"].unsqueeze(-1)  # [batch_size, 1]

        # Get success labels for all trajectories
        success_labels_ref = inputs["success_labels_ref"]  # [batch_size, seq_len]
        success_labels_sim = inputs["success_labels_sim"]  # [batch_size, seq_len]
        success_labels_diff = inputs["success_labels_diff"]  # [batch_size, seq_len]

        if self.config.model.train_progress_head and self.config.training.predict_sim_progress:
            # Get progress logits for both comparisons
            progress_logits_ref_sim = model_outputs_ref_sim.progress_logits
            progress_logits_ref_diff = model_outputs_ref_diff.progress_logits
            progress_pred_ref_sim_A = progress_logits_ref_sim["A"]  # [batch_size, seq_len]
            progress_pred_ref_sim_B = progress_logits_ref_sim.get("B")  # [batch_size, seq_len] or None
            progress_pred_ref_diff_A = progress_logits_ref_diff["A"]  # [batch_size, seq_len]
            progress_pred_ref_diff_B = progress_logits_ref_diff.get("B")  # [batch_size, seq_len] or None

            # For ref_sim: select trajectory A (ref) or B (sim) based on random indicator
            # progress_pred_traj_ref_sim[i] == 1.0 means use A (ref), 0.0 means use B (sim)
            ref_sim_use_A = (progress_pred_traj_ref_sim == 1.0).float().unsqueeze(-1)  # [batch_size, 1]
            target_progress_ref_sim = ref_sim_use_A * target_progress_ref + (1 - ref_sim_use_A) * target_progress_sim
            target_progress_ref_sim_mask = (
                ref_sim_use_A * target_progress_ref_mask + (1 - ref_sim_use_A) * target_progress_sim_mask
            )
            progress_pred_ref_sim = ref_sim_use_A * progress_pred_ref_sim_A
            if progress_pred_ref_sim_B is not None:
                progress_pred_ref_sim = progress_pred_ref_sim + (1 - ref_sim_use_A) * progress_pred_ref_sim_B

            # For ref_diff: select trajectory A (ref) or B (diff) based on random indicator
            # progress_pred_traj_ref_diff[i] == 1.0 means use A (ref), 0.0 means use B (diff)
            ref_diff_use_A = (progress_pred_traj_ref_diff == 1.0).float().unsqueeze(-1)  # [batch_size, 1]
            target_progress_ref_diff = (
                ref_diff_use_A * target_progress_ref + (1 - ref_diff_use_A) * target_progress_diff
            )
            target_progress_ref_diff_mask = (
                ref_diff_use_A * target_progress_ref_mask + (1 - ref_diff_use_A) * target_progress_diff_mask
            )
            progress_pred_ref_diff = ref_diff_use_A * progress_pred_ref_diff_A
            if progress_pred_ref_diff_B is not None:
                progress_pred_ref_diff = progress_pred_ref_diff + (1 - ref_diff_use_A) * progress_pred_ref_diff_B

            # Compute progress loss for ref_sim
            progress_loss_ref_sim, spearman_corr_ref_sim, progress_metrics_ref_sim = self._compute_progress_loss_helper(
                progress_pred_ref_sim,
                target_progress_ref_sim,
                target_progress_ref_sim_mask,
            )

            # Compute progress loss for ref_diff
            progress_loss_ref_diff, spearman_corr_ref_diff, progress_metrics_ref_diff = (
                self._compute_progress_loss_helper(
                    progress_pred_ref_diff,
                    target_progress_ref_diff,
                    target_progress_ref_diff_mask,
                )
            )

            # Sum the progress losses
            total_progress_loss = progress_loss_ref_sim + progress_loss_ref_diff
            final_loss = similarity_loss + total_progress_loss

        if self.config.model.train_success_head:
            # Get success logits for both comparisons
            success_logits_ref_sim = model_outputs_ref_sim.success_logits
            success_logits_ref_diff = model_outputs_ref_diff.success_logits
            success_pred_ref_sim_A = success_logits_ref_sim["A"]  # [batch_size, seq_len]
            success_pred_ref_sim_B = success_logits_ref_sim.get("B")  # [batch_size, seq_len] or None
            success_pred_ref_diff_A = success_logits_ref_diff["A"]  # [batch_size, seq_len]
            success_pred_ref_diff_B = success_logits_ref_diff.get("B")  # [batch_size, seq_len] or None

            # For ref_sim: select trajectory A (ref) or B (sim) based on random indicator
            # success_pred_traj_ref_sim[i] == 1.0 means use A (ref), 0.0 means use B (sim)
            ref_sim_use_A_success = (success_pred_traj_ref_sim == 1.0).float().unsqueeze(-1)  # [batch_size, 1]
            target_progress_ref_sim_success = (
                ref_sim_use_A_success * target_progress_ref + (1 - ref_sim_use_A_success) * target_progress_sim
            )
            success_labels_ref_sim = (
                ref_sim_use_A_success * success_labels_ref + (1 - ref_sim_use_A_success) * success_labels_sim
            )
            target_progress_ref_sim_mask_success = (
                ref_sim_use_A_success * target_progress_ref_mask
                + (1 - ref_sim_use_A_success) * target_progress_sim_mask
            )
            success_pred_ref_sim = ref_sim_use_A_success * success_pred_ref_sim_A
            if success_pred_ref_sim_B is not None:
                success_pred_ref_sim = success_pred_ref_sim + (1 - ref_sim_use_A_success) * success_pred_ref_sim_B

            # For ref_diff: select trajectory A (ref) or B (diff) based on random indicator
            # success_pred_traj_ref_diff[i] == 1.0 means use A (ref), 0.0 means use B (diff)
            ref_diff_use_A_success = (success_pred_traj_ref_diff == 1.0).float().unsqueeze(-1)  # [batch_size, 1]
            target_progress_ref_diff_success = (
                ref_diff_use_A_success * target_progress_ref + (1 - ref_diff_use_A_success) * target_progress_diff
            )
            success_labels_ref_diff = (
                ref_diff_use_A_success * success_labels_ref + (1 - ref_diff_use_A_success) * success_labels_diff
            )
            target_progress_ref_diff_mask_success = (
                ref_diff_use_A_success * target_progress_ref_mask
                + (1 - ref_diff_use_A_success) * target_progress_diff_mask
            )
            success_pred_ref_diff = ref_diff_use_A_success * success_pred_ref_diff_A
            if success_pred_ref_diff_B is not None:
                success_pred_ref_diff = success_pred_ref_diff + (1 - ref_diff_use_A_success) * success_pred_ref_diff_B

            # Compute success loss for ref_sim
            success_loss_ref_sim, success_accuracy_ref_sim, success_auprc_ref_sim, success_metrics_ref_sim = (
                self._compute_success_loss_helper(
                    success_pred_ref_sim,
                    target_progress_ref_sim_success,
                    success_labels_ref_sim,
                    progress_loss_mask=target_progress_ref_sim_mask_success,
                )
            )

            # Compute success loss for ref_diff
            success_loss_ref_diff, success_accuracy_ref_diff, success_auprc_ref_diff, success_metrics_ref_diff = (
                self._compute_success_loss_helper(
                    success_pred_ref_diff,
                    target_progress_ref_diff_success,
                    success_labels_ref_diff,
                    progress_loss_mask=target_progress_ref_diff_mask_success,
                )
            )

            # Sum the success losses
            total_success_loss = success_loss_ref_sim + success_loss_ref_diff
            success_accuracy = (success_accuracy_ref_sim + success_accuracy_ref_diff) / 2.0
            success_auprc = (success_auprc_ref_sim + success_auprc_ref_diff) / 2.0
            final_loss = final_loss + total_success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            logger.warning(f"NaN detected in similarity loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}
            prefix = "train" if training else "eval"

            # Compute similarity ranking accuracy
            # If score_ref_sim > score_ref_diff, model correctly ranks sim as more similar
            similarity_correct = (score_ref_sim > score_ref_diff).float()
            similarity_accuracy = similarity_correct.mean()
            avg_similarity_margin = similarity_margin.mean()

            data_gen_strategy = inputs["data_gen_strategy"]

            # Prepare metrics for stratification
            stratified_metrics = {
                "sim_acc": similarity_correct,
                "sim_loss": similarity_loss_all,
                "sim_margin": similarity_margin,
            }

            self._add_stratified_metrics(
                outputs_dict,
                prefix,
                data_gen_strategy,
                inputs["data_source"],
                stratified_metrics,
            )

            # Add main metrics
            outputs_dict.update({
                f"{prefix}/similarity_loss": similarity_loss.item(),
                f"{prefix}/similarity_acc": similarity_accuracy.item(),
                f"{prefix}/similarity_margin": avg_similarity_margin.item(),
            })

            # Add progress loss metrics if computed
            if self.config.model.train_progress_head and self.config.training.predict_sim_progress:
                outputs_dict.update({
                    f"{prefix}/sim_prog_loss": total_progress_loss.item(),
                    f"{prefix}/sim_prog_loss_ref_sim": progress_loss_ref_sim.item(),
                    f"{prefix}/sim_prog_loss_ref_diff": progress_loss_ref_diff.item(),
                    f"{prefix}/sim_prog_spearman_corr": (spearman_corr_ref_sim + spearman_corr_ref_diff).item() / 2.0,
                })

                # Combine metrics from both ref_sim and ref_diff for stratification
                # Average the metrics across both comparisons
                combined_spearman_corr = (
                    progress_metrics_ref_sim["masked_spearman_corr"] + progress_metrics_ref_diff["masked_spearman_corr"]
                ) / 2.0
                combined_prog_loss = (
                    progress_metrics_ref_sim["masked_loss"] + progress_metrics_ref_diff["masked_loss"]
                ) / 2.0

                stratified_progress_metrics = {
                    "spearman_corr": combined_spearman_corr,
                    "prog_loss": combined_prog_loss,
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["data_gen_strategy"],
                    inputs["data_source"],
                    stratified_progress_metrics,
                )

            # Add success loss metrics if computed
            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/sim_success_loss": total_success_loss.item(),
                    f"{prefix}/sim_success_loss_ref_sim": success_loss_ref_sim.item(),
                    f"{prefix}/sim_success_loss_ref_diff": success_loss_ref_diff.item(),
                    f"{prefix}/sim_success_accuracy": success_accuracy.item(),
                    f"{prefix}/sim_success_auprc": success_auprc.item(),
                })

                # Combine metrics from both ref_sim and ref_diff for stratification
                # Average the metrics across both comparisons
                combined_success_loss = (
                    success_metrics_ref_sim["masked_loss"] + success_metrics_ref_diff["masked_loss"]
                ) / 2.0
                combined_success_acc = (
                    success_metrics_ref_sim["masked_correct"] + success_metrics_ref_diff["masked_correct"]
                ) / 2.0

                stratified_success_metrics = {
                    "success_loss": combined_success_loss,
                    "success_acc": combined_success_acc,
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["data_gen_strategy"],
                    inputs["data_source"],
                    stratified_success_metrics,
                )

            return final_loss, outputs_dict

        return final_loss
