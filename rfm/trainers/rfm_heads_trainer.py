import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from rfm.utils.logger import Logger

import copy
import wandb
from rfm.utils.distributed import is_rank_0, rank_0_print
from rfm.utils.timer import _timer
from rfm.utils.metrics import compute_spearman_correlation
from rfm.utils.setup_utils import setup_dataset, setup_batch_collator, setup_custom_eval_dataset
from torch.utils.data import DataLoader
from rfm.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING
from evals.compile_results import compute_eval_metrics
from rfm.data.datasets.helpers import load_dataset_success_percent
import torch.distributed as dist
from rfm.data.datasets.base import resolve_dataset_keys
from rfm.utils.distributed import banner
from rfm.models.utils import ModelOutput


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
                rank_0_print(f"Warning: NaN detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Check for infinity values
            if torch.isinf(tensor_val).any():
                rank_0_print(f"Warning: Infinity detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Use accelerator's reduce method - all processes participate
            reduced_val = accelerator.reduce(tensor_val, reduction=aggregate_method)

            # Final check for NaN in reduced result
            if torch.isnan(reduced_val).any():
                rank_0_print(f"Warning: NaN in reduced result for metric '{key}', using fallback")
                result_metrics[key] = 0.0
            else:
                result_metrics[key] = reduced_val.item()

        except Exception as metric_error:
            # If individual metric fails, keep original value (or 0.0 if missing)
            rank_0_print(f"Warning: Failed to reduce metric '{key}': {metric_error}")
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

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(
                log_to=self.config.logging.log_to,
                output_dir=getattr(self.args, "output_dir", "./logs"),
                is_main_process=is_rank_0(),
            )

        # Load dataset-specific success perfect percentage
        cutoff_file_path = config.data.dataset_success_cutoff_file
        self.dataset_success_percent = load_dataset_success_percent(cutoff_file_path)

        rank_0_print(f"Dataset success perfect percentage:")
        for k, v in self.dataset_success_percent.items():
            rank_0_print(f"  {k} - {v}")
        rank_0_print(f"=" * 100)
        rank_0_print(f"DDP find_unused_parameters: {getattr(self.args, 'ddp_find_unused_parameters', 'N/A')}")

    def _post_checkpoint_load_reset(self):
        """
        Reset model and optimizer state after loading from checkpoint.
        This addresses issues where checkpoint loading can leave stale gradients
        or computational graph state that causes crashes during training.
        """
        rank_0_print("Performing post-checkpoint load reset...")

        # Ensure model is in training mode
        self.model.train()

        # Clear any cached gradients or computational graph state
        # NOTE: We don't clear optimizer.state or param_groups as that breaks the lr_scheduler
        try:
            # Zero out any existing gradients
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            rank_0_print(f"Warning: Could not clear gradients: {e}")

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        rank_0_print("Post-checkpoint load reset complete")

    def _log_rank_shape(self, label: str, value) -> None:
        """Utility to print tensor/list shapes per rank for debugging gather issues."""
        if not hasattr(self, "accelerator"):
            return
        rank = getattr(self.accelerator, "process_index", None)
        if rank is None and dist.is_initialized():
            rank = dist.get_rank()
        if rank is None:
            rank = 0

        if torch.is_tensor(value):
            info = f"tensor shape={tuple(value.shape)}"
        elif isinstance(value, (list, tuple)):
            info = f"{type(value).__name__} len={len(value)}"
        elif value is None:
            info = "None"
        else:
            info = f"{type(value).__name__}"

        print(f"[Rank {rank}] {label}: {info}", flush=True)

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
            rank_0_print(f"Warning: Could not get learning rate: {e}")
            return self.args.learning_rate

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Override train method to perform post-checkpoint reset.
        """
        # If resuming from checkpoint, set flag for reset in first training step
        if resume_from_checkpoint is not None:
            rank_0_print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self._just_resumed_from_checkpoint = True

        # Call parent train method
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        # Check if we just resumed from checkpoint (first step after resume)
        if hasattr(self, "_just_resumed_from_checkpoint") and self._just_resumed_from_checkpoint:
            self._post_checkpoint_load_reset()
            self._just_resumed_from_checkpoint = False

        self.timing_raw = {}

        # Initialize log_metadata
        self.log_metadata = {}

        # Safety check: ensure model is in training mode and gradients are properly set up
        if not model.training:
            rank_0_print("Warning: Model not in training mode, setting to train mode")
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

        # Adding more granular counting for the data strategies
        if num_preferences > 0 and preference_inputs:
            rejected_data_gen_strategy = preference_inputs["rejected_data_gen_strategy"]
            if isinstance(rejected_data_gen_strategy, list) and len(rejected_data_gen_strategy) > 0:
                for s in rejected_data_gen_strategy:
                    if s == "rewind_same_task":
                        self.global_metadata["pref_num_trajs_rewind"] += 1
                    elif s == "suboptimal_same_task":
                        self.global_metadata["pref_num_trajs_same_task_subopt"] += 1
                    elif s == "different_task":
                        self.global_metadata["pref_num_trajs_diff_task"] += 1

            data_sources = preference_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        if num_progress > 0 and progress_inputs:
            data_gen_strategy = progress_inputs["data_gen_strategy"]
            if isinstance(data_gen_strategy, list) and len(data_gen_strategy) > 0:
                for s in data_gen_strategy:
                    if s == "successful":
                        self.global_metadata["prog_num_trajs_succ"] += 1
                    elif s == "rewind_same_task":
                        self.global_metadata["prog_num_trajs_rewind"] += 1
                    elif s == "different_task":
                        self.global_metadata["prog_num_trajs_diff_task"] += 1

            data_sources = progress_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        if num_similarities > 0 and similarity_inputs:
            data_gen_strategy = similarity_inputs["data_gen_strategy"]
            if isinstance(data_gen_strategy, list) and len(data_gen_strategy) > 0:
                for s in data_gen_strategy:
                    if s == "rewind_same_task":
                        self.global_metadata["sim_num_trajs_rewind"] += 1
                    elif s == "suboptimal_same_task":
                        self.global_metadata["sim_num_trajs_same_task_subopt"] += 1
                    elif s == "paired_human_robot":
                        self.global_metadata["sim_num_trajs_paired_hr"] += 1

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

        self._update_resample_attempt_metrics(inputs)

        # Log custom losses at specified intervals (using our custom logger only)
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_metadata()

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

        # Use local metrics (no aggregation needed for individual GPU metrics)
        log_metadata = reduce_metrics_with_accelerate(self.log_metadata, self.accelerator, aggregate_method="mean")

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
            **self.timing_raw,
            **log_metadata,
        }

        # Log global metadata
        global_metadata = reduce_metrics_with_accelerate(self.global_metadata, self.accelerator, aggregate_method="sum")
        log_global = {f"counts/{key}": global_metadata[key] for key in global_metadata}
        log_data.update(log_global)

        # Log optimizer and gradient statistics
        optim_stats = self._get_optimizer_stats()
        log_data.update(optim_stats)

        # make sure values are floats so they are loggable into wandb reports
        log_data = {k: float(v) for k, v in log_data.items()}

        self.logger.log_scalars(log_data, step=self.state.global_step)

        if is_rank_0():
            rank_0_print(f"Step {self.state.global_step}:")
            rank_0_print("-" * 50)
            for key in log_global:
                rank_0_print(f"  {key}: {log_global[key]}")

            rounded_times = {k: round(v, 2) for k, v in self.timing_raw.items()}
            rank_0_print(f"Timing raw: {rounded_times}")

            # Log optimizer stats to console
            if optim_stats:
                rank_0_print(f"Optimizer stats: {optim_stats}")

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
            persistent_workers=self.args.dataloader_persistent_workers,
            worker_init_fn=seed_worker,
        )
        return self.accelerator.prepare(dl)

    def _run_custom_evaluations(self):
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

        for eval_type in eval_types:
            rank_0_print(f"Running evaluation for: {eval_type}")

            datasets = getattr(self.config.custom_eval, eval_type)
            eval_datasets_name = resolve_dataset_keys(datasets, split="eval")

            for eval_dataset in eval_datasets_name:
                eval_cfg = copy.deepcopy(self.config.data)
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

                dataset = setup_custom_eval_dataset(
                    eval_cfg, sampler_type=eval_type, is_eval=True, verbose=False, **kwargs
                )
                dataloader = self._make_eval_dataloader(dataset)

                # Ensure model is in eval mode and clear any gradient buffers
                self.model.eval()
                # Explicitly zero any gradients that might exist (shouldn't, but safety measure)
                if hasattr(self, "optimizer") and self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                eval_results = []

                # Initialize success_auprc to None for all eval types
                success_auprc = None

                for batch in tqdm(
                    dataloader,
                    desc=f"Running {eval_type}, ds: {eval_dataset}, batch size: {self.config.training.per_device_eval_batch_size}",
                ):
                    batch = self._prepare_inputs(batch)

                    if eval_type in [
                        "reward_alignment",
                        "policy_ranking",
                        "confusion_matrix",
                    ]:
                        progress_samples = batch["progress_inputs"]
                        with torch.no_grad():
                            outputs, _ = self.forward_model(self.model, progress_samples, sample_type="progress")

                        progress_logits = outputs.progress_logits
                        progress_pred = progress_logits["A"]

                        if isinstance(progress_pred, list):
                            if isinstance(progress_pred[0], torch.Tensor):
                                progress_pred = torch.stack(progress_pred)
                            else:
                                progress_pred = torch.tensor(progress_pred, device=self.accelerator.device)

                        # self._log_rank_shape("progress_pred_local", progress_pred)
                        # self._log_rank_shape("target_progress_local", progress_samples["target_progress"])

                        # Gather everything
                        progress_pred = self.accelerator.gather_for_metrics(progress_pred)
                        target_progress = self.accelerator.gather_for_metrics(progress_samples["target_progress"])
                        # self._log_rank_shape("progress_pred_gathered", progress_pred)
                        # self._log_rank_shape("target_progress_gathered", target_progress)

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
                        gathered_metadata_dict = self._truncate_metadata_lists(
                            gathered_metadata_dict, num_progress_samples
                        )

                        # Handle success predictions if needed
                        success_pred_gathered = None
                        success_probs_gathered = None
                        # success_auprc is initialized before the loop, only update it here if computing
                        if self.config.model.train_success_head:
                            success_pred = outputs.success_logits["A"]
                            if isinstance(success_pred, list):
                                success_pred = (
                                    torch.stack(success_pred)
                                    if isinstance(success_pred[0], torch.Tensor)
                                    else torch.tensor(success_pred, device=self.accelerator.device)
                                )
                            success_probs = torch.sigmoid(success_pred)
                            # self._log_rank_shape("success_probs_local", success_probs)
                            success_binary = (success_probs > 0.5).float()
                            success_pred_gathered = self.accelerator.gather_for_metrics(success_binary)
                            success_probs_gathered = self.accelerator.gather_for_metrics(success_probs)
                            # self._log_rank_shape("success_pred_gathered", success_pred_gathered)
                            # self._log_rank_shape("success_probs_gathered", success_probs_gathered)

                            # Compute success labels from target_progress for AUPRC
                            # Use the same logic as in _compute_success_loss_helper
                            min_success = self.config.data.min_success
                            max_success_default = self.config.data.max_success

                            # Handle Qwen downsampling if needed
                            target_progress_for_success = target_progress
                            if "Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image:
                                target_progress_for_success = target_progress[:, ::2]
                            # self._log_rank_shape("target_progress_for_success", target_progress_for_success)

                            # Compute per-sample max_success thresholds
                            data_sources_list = gathered_metadata_dict["data_source"]
                            max_success_tensor = torch.tensor(
                                [
                                    self.dataset_success_percent.get(
                                        ds if isinstance(ds, str) else str(ds), max_success_default
                                    )
                                    for ds in data_sources_list
                                ],
                                dtype=target_progress_for_success.dtype,
                                device=target_progress_for_success.device,
                            )

                            # Generate success labels: 1.0 for success, 0.0 for failure
                            success_labels = (target_progress_for_success > max_success_tensor.unsqueeze(1)).float()

                            # Ensure shapes match - target_progress must be a sequence to compute frame-level success
                            if success_probs_gathered.shape != target_progress_for_success.shape:
                                raise ValueError(
                                    f"Shape mismatch for AUPRC computation: "
                                    f"success_probs_gathered.shape={success_probs_gathered.shape}, "
                                    f"target_progress_for_success.shape={target_progress_for_success.shape}. "
                                    f"target_progress must be a sequence (frame-level) to compute frame-level success AUPRC. "
                                    f"If target_progress is scalar [batch, 1], frame-level success cannot be computed."
                                )

                            # Compute AUPRC across all valid frames
                            # Flatten tensors for AUPRC computation
                            success_probs_flat = success_probs_gathered.flatten()
                            success_labels_flat = success_labels.flatten()

                            if success_probs_flat.numel() > 0 and len(torch.unique(success_labels_flat)) > 1:
                                auprc_value = average_precision_score(
                                    success_labels_flat.detach().cpu().float().numpy(),
                                    success_probs_flat.detach().cpu().float().numpy(),
                                )
                                success_auprc = float(auprc_value)
                            else:
                                success_auprc = 0.0

                            # Clean up intermediate tensors
                            del (
                                success_pred,
                                success_binary,
                                success_probs,
                                target_progress_for_success,
                                max_success_tensor,
                                success_labels,
                            )

                        # Build eval_results on all processes for compute_eval_metrics
                        for i in range(len(progress_pred)):
                            metadata = gathered_metadata_dict["metadata"][i]
                            sample_result = {
                                "task": gathered_metadata_dict["task"][i],
                                "target_progress": target_progress[i].detach().to(dtype=torch.float32).cpu().numpy(),
                                "progress_pred": progress_pred[i].detach().to(dtype=torch.float32).cpu().numpy(),
                                "data_source": gathered_metadata_dict["data_source"][i],
                                "data_gen_strategy": gathered_metadata_dict["data_gen_strategy"][i],
                                "quality_label": gathered_metadata_dict["quality_labels"][i],
                                "metadata": metadata,
                                "id": metadata["id"],
                                "video_path": metadata["video_path"],
                                "partial_success": gathered_metadata_dict["partial_success"][i],
                            }
                            if success_pred_gathered is not None:
                                sample_result["success_pred"] = (
                                    success_pred_gathered[i].detach().to(dtype=torch.float32).cpu().numpy()
                                )
                            eval_results.append(sample_result)

                        # Clean up gathered tensors and metadata after building results
                        del progress_pred, target_progress, gathered_metadata_dict
                        if success_pred_gathered is not None:
                            del success_pred_gathered
                        if success_probs_gathered is not None:
                            del success_probs_gathered

                    elif eval_type == "quality_preference" or eval_type == "quality_preference_roboarena":
                        # Process preference samples for quality_preference evaluation
                        preference_samples = batch["preference_inputs"]
                        with torch.no_grad():
                            outputs, _ = self.forward_model(self.model, preference_samples, sample_type="preference")
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
                        gathered_pref_metadata = self._truncate_metadata_lists(
                            gathered_pref_metadata, num_pref_samples
                        )
                        gathered_task = gathered_pref_metadata["task"]
                        gathered_data_source = gathered_pref_metadata["data_source"]
                        gathered_chosen_data_gen_strategy = gathered_pref_metadata["chosen_data_gen_strategy"]
                        gathered_rejected_data_gen_strategy = gathered_pref_metadata["rejected_data_gen_strategy"]
                        gathered_metadata = gathered_pref_metadata["metadata"]

                        # Build eval_results on all processes for compute_eval_metrics
                        for i in range(len(pref_logits)):
                            if pref_logits[i] is None:
                                continue
                            sample_result = {
                                "task": gathered_task[i],
                                "preference_pred": pref_logits[i].detach().to(dtype=torch.float32).cpu().numpy(),
                                "preference_labels": preference_labels[i]
                                .detach()
                                .to(dtype=torch.float32)
                                .cpu()
                                .numpy(),
                                "data_source": gathered_data_source[i],
                                "chosen_data_gen_strategy": gathered_chosen_data_gen_strategy[i],
                                "rejected_data_gen_strategy": gathered_rejected_data_gen_strategy[i],
                                "metadata": gathered_metadata[i],
                            }
                            eval_results.append(sample_result)

                        # Clean up gathered tensors and metadata after building results
                        del pref_logits, preference_labels
                        del gathered_task, gathered_data_source, gathered_chosen_data_gen_strategy
                        del gathered_rejected_data_gen_strategy, gathered_metadata

                    elif eval_type == "similarity_score":
                        # Process similarity samples for similarity_score evaluation
                        similarity_samples = batch["similarity_inputs"]
                        with torch.no_grad():
                            outputs, _ = self.forward_model(self.model, similarity_samples, sample_type="similarity")
                        sim_logits = outputs.sim_logits

                        # Gather predictions across all ranks
                        sim_logits = self.accelerator.gather_for_metrics(sim_logits)

                        # Gather non-tensor metadata using helper (handles optional/None entries)
                        gathered_sim_metadata = self._gather_metadata_fields(
                            similarity_samples,
                            ["task", "data_source", "data_gen_strategy", "metadata"],
                        )
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
                        num_samples = len(sim_logits) // 2
                        for i in range(num_samples):
                            ref_sim_idx = i * 2
                            ref_diff_idx = i * 2 + 1

                            if ref_sim_idx >= len(sim_logits) or ref_diff_idx >= len(sim_logits):
                                continue

                            # Metadata is indexed by sample index (i), not batched index
                            sample_result = {
                                "task": gathered_task[i] if i < len(gathered_task) else None,
                                "sim_score_ref_sim": sim_logits[ref_sim_idx]
                                .detach()
                                .to(dtype=torch.float32)
                                .cpu()
                                .numpy(),
                                "sim_score_ref_diff": sim_logits[ref_diff_idx]
                                .detach()
                                .to(dtype=torch.float32)
                                .cpu()
                                .numpy(),
                                "data_source": gathered_data_source[i] if i < len(gathered_data_source) else None,
                                "data_gen_strategy": gathered_data_gen_strategy[i]
                                if i < len(gathered_data_gen_strategy)
                                else None,
                                "metadata": gathered_metadata[i] if i < len(gathered_metadata) else None,
                            }
                            eval_results.append(sample_result)

                        # Clean up gathered tensors and metadata after building results
                        del sim_logits
                        del gathered_task, gathered_data_source, gathered_data_gen_strategy, gathered_metadata

                    # Clean up batch tensors and free memory after each batch
                    # This is critical for VQA with generation to prevent OOM
                    del batch, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Compute metrics on all processes
                # Initialize variables to None to ensure they exist for cleanup
                plots = None
                video_frames_list = None
                task_groups = None
                task_details = None
                confusion_plot = None
                confusion_matrix = None

                if eval_type == "reward_alignment":
                    eval_metrics, plots, video_frames_list = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                elif eval_type == "policy_ranking":
                    # create task groups from eval_results (handles roboarena automatically)
                    eval_metrics, task_groups, task_details = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                elif eval_type == "confusion_matrix":
                    confusion_plot, confusion_matrix = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                elif eval_type == "quality_preference" or eval_type == "quality_preference_roboarena":
                    # quality_preference returns metrics, task_groups, and task_details
                    eval_metrics, task_groups, task_details = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                elif eval_type == "similarity_score":
                    # similarity_score returns metrics, task_groups, and task_details
                    eval_metrics, task_groups, task_details = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                else:
                    raise ValueError(f"Unsupported eval type: {eval_type}")

                # Add success_auprc to metrics if computed (for eval types with success predictions)
                if success_auprc is not None and eval_type in [
                    "reward_alignment",
                    "policy_ranking",
                    "confusion_matrix",
                ]:
                    if not isinstance(eval_metrics, dict):
                        eval_metrics = {}
                    eval_metrics["success_auprc"] = success_auprc

                # Store metrics for all processes
                # For similarity_score, eval_dataset is a list, so use the first element for name mapping
                dataset_for_mapping = eval_dataset[0] if isinstance(eval_dataset, list) else eval_dataset
                ds_name = DS_SHORT_NAME_MAPPING.get(dataset_for_mapping, dataset_for_mapping)
                metrics[ds_name][eval_type] = eval_metrics

                # Only log and visualize on main process
                if self.accelerator.is_main_process:
                    banner(
                        "Completed evaluation",
                        f"{eval_type} evaluation: {len(eval_results)} samples",
                        "Metrics",
                        f"{metrics[ds_name][eval_type]}",
                        inner_padding=1,
                    )

                    # Create wandb tables and log visualizations
                    if eval_type == "reward_alignment":
                        # Build rows of (video, figure)
                        rows = []
                        for plot, frames in zip(plots, video_frames_list):
                            if frames is not None:
                                rows.append((frames, plot))
                        if rows and self.logger.enabled("wandb"):
                            self.logger.log_video_table(
                                f"{ds_name}/reward_alignment_samples",
                                videos_and_figures=rows,
                                columns=["video", "progress_plot"],
                                step=self.state.global_step,
                            )
                        # For tensorboard (no table support), log each video and its figure separately
                        if self.logger.enabled("tensorboard"):
                            for idx, frames in enumerate(video_frames_list):
                                if frames is not None:
                                    self.logger.log_video(
                                        f"{ds_name}/reward_alignment_video/{idx}",
                                        frames,
                                        fps=2,
                                        step=self.state.global_step,
                                    )
                            for idx, plot in enumerate(plots):
                                self.logger.log_figure(
                                    f"{ds_name}/reward_alignment_plot/{idx}", plot, step=self.state.global_step
                                )
                        # Close all plots to avoid accumulating open figures
                        for plot in plots:
                            plt.close(plot)

                        # Explicitly delete to free memory and set to None for outer cleanup
                        del plots, video_frames_list, rows
                        plots = None
                        video_frames_list = None

                    elif eval_type == "policy_ranking":
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
                                partial_successes = [round(t["partial_success"], 3) for t in group]
                                predicted_rewards = [round(t["final_predicted_reward"], 3) for t in group]
                                spearman = task_details.get(task, {}).get("spearman", 0.0)
                                data.append([
                                    task,
                                    f"partial:{partial_successes}",
                                    f"predicted:{predicted_rewards}",
                                    round(spearman, 3),
                                ])
                            columns = ["task", "partial_successes", "predicted_rewards", "spearman"]
                            table_name = f"{ds_name}/policy_ranking_roboarena_samples"
                        else:
                            # Standard policy ranking visualization: show quality labels and rewards
                            for task, group in task_groups.items():
                                quality_to_rews = collections.defaultdict(list)
                                for t in group:
                                    quality_to_rews[t["quality_label"]].append(round(t["final_reward"], 2))
                                quality_to_rews = ",".join([f"{q}:{r}" for q, r in quality_to_rews.items()])
                                data.append([task, quality_to_rews])
                            columns = ["task", "quality_and_rews"]
                            table_name = f"{ds_name}/policy_ranking_samples"

                        self.logger.log_table(
                            table_name,
                            data=data,
                            columns=columns,
                            step=self.state.global_step,
                        )
                        del data, task_groups, task_details
                        task_groups = None
                        task_details = None

                    elif eval_type == "quality_preference" or eval_type == "quality_preference_roboarena":
                        # Check if this is roboarena by checking if task_details have partial_success_accuracies
                        is_roboarena = False
                        if task_details:
                            first_task_details = next(iter(task_details.values()))
                            if "partial_success_accuracies" in first_task_details:
                                is_roboarena = True

                        data = []
                        if is_roboarena:
                            # RoboArena visualization: show partial_success accuracies
                            for task, details in task_details.items():
                                task_acc = details["preference_accuracy"]
                                partial_accs = details.get("partial_success_accuracies", {})
                                partial_accs_str = ",".join([f"{k}:{round(v, 3)}" for k, v in partial_accs.items()])
                                num_correct = details["num_correct"]
                                num_total = details["num_total"]
                                data.append([
                                    task,
                                    round(task_acc, 3),
                                    partial_accs_str if partial_accs_str else "N/A",
                                    f"{num_correct}/{num_total}",
                                ])
                            columns = ["task", "preference_accuracy", "partial_success_accuracies", "num_correct/total"]
                            table_name = f"{ds_name}/quality_preference_roboarena_samples"
                        else:
                            # Standard quality_preference visualization: show quality label accuracies
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
                            table_name = f"{ds_name}/quality_preference_samples"

                        self.logger.log_table(
                            table_name,
                            data=data,
                            columns=columns,
                            step=self.state.global_step,
                        )
                        del data, task_groups, task_details
                        task_groups = None
                        task_details = None

                    elif eval_type == "confusion_matrix":
                        self.logger.log_figure(
                            f"{ds_name}/confusion_matrix", confusion_plot, step=self.state.global_step
                        )
                        plt.close(confusion_plot)
                        del confusion_plot, confusion_matrix
                        confusion_plot = None
                        confusion_matrix = None

                    elif eval_type == "similarity_score":
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
                            f"{ds_name}/similarity_score_samples",
                            data=data,
                            columns=columns,
                            step=self.state.global_step,
                        )
                        del data, task_groups, task_details
                        task_groups = None
                        task_details = None

                # Clean up after each dataset to prevent memory accumulation
                # Clean up eval-specific outputs (now properly scoped)
                if plots is not None:
                    del plots
                if video_frames_list is not None:
                    del video_frames_list
                if task_groups is not None:
                    del task_groups
                if task_details is not None:
                    del task_details
                if confusion_plot is not None:
                    del confusion_plot
                if confusion_matrix is not None:
                    del confusion_matrix

                # Clean up dataset, dataloader, and eval_results
                del dataset, dataloader, eval_results

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc

                gc.collect()

        # Prepare metrics for callbacks (all processes)
        callback_metrics = {}
        for ds_name, eval_type_metric in metrics.items():
            for eval_type, metric in eval_type_metric.items():
                eval_type_short = EVAL_TYPE_SHORT[eval_type]
                # Add to callback metrics
                for k, v in metric.items():
                    if isinstance(v, (int, float)):
                        metric_name = f"custom_eval/{eval_type_short}_{k}_{ds_name}"
                        callback_metrics[metric_name] = v

        # Prepare wandb metrics and log (only on main process)
        if self.accelerator.is_main_process:
            to_log = {}
            for ds_name, eval_type_metric in metrics.items():
                for eval_type, metric in eval_type_metric.items():
                    eval_type_short = EVAL_TYPE_SHORT[eval_type]
                    for k, v in metric.items():
                        if isinstance(v, (int, float)):
                            metric_name = f"custom_eval/{eval_type_short}_{k}_{ds_name}"
                            to_log[metric_name] = float(v)
            self.logger.log_scalars(to_log, step=self.state.global_step)

        banner("Finished running custom evaluations!")

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

        return callback_metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
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

            # Log metrics
            if is_rank_0():
                banner("Custom RFM Evaluation Results (Aggregated)", inner_padding=1)
                for key, value in metrics.items():
                    rank_0_print(f"{key}: {value:.6f}")
                rank_0_print("=" * 50)

            # Also log to wandb if available and configured (only on rank 0)
            self.logger.log_scalars(metrics, step=self.state.global_step)

        # Run the custom evaluations
        custom_eval_should_run = (
            self.config.training.custom_eval_steps
            and self.state.global_step % self.config.training.custom_eval_steps == 0
        )
        if custom_eval_should_run:
            custom_metrics = self._run_custom_evaluations()
            metrics.update(custom_metrics)

            # to trigger the callback handler
            # self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

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

        # Set static graph for DDP on first training step to handle multiple forward passes
        # This is necessary because similarity loss does 2 forward passes (ref_sim and ref_diff)
        if (
            training
            and not self._ddp_static_graph_set
            and getattr(self.accelerator.gradient_state, "sync_gradients", True)
            and hasattr(model, "module")
        ):
            if hasattr(model.module, "_set_static_graph"):
                rank_0_print("Setting DDP static graph mode for multiple forward passes")
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, "_set_static_graph"):
                rank_0_print("Setting DDP static graph mode for multiple forward passes")
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

        # Check for NaN in total loss before returning
        if torch.isnan(total_loss).any():
            rank_0_print(f"Warning: NaN detected in total_loss, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {**log_metadata, "total_loss": total_loss.item()}
            return total_loss, extra_info

        return total_loss

    def _compute_success_loss_helper(
        self,
        success_logits,
        target_progress,
        progress_loss_mask=None,
        data_source=None,
        aggregate: bool = False,
    ):
        """
        Helper function to compute success prediction loss.

        Computes binary cross-entropy loss for frames with:
        - progress < min_success (label=0, failure)
        - progress > max_success (label=1, success)
        - ignores frames in between

        Args:
            success_logits: Success prediction logits (can be tensor or list of tensors)
            target_progress: Target progress tensors (can be tensor or list of tensors)
            progress_loss_mask: Per-sample mask tensor of shape (batch_size,) with 1.0 for samples
                where we should compute progress/success loss (e.g., successful, rewound, different_task)
            data_source: Dataset source information for threshold lookup
            aggregate: Whether to return the mean of the losses and accuracies

        Returns:
            tuple: (success_loss, success_accuracy, success_auprc) if aggregate=True
                   (masked_loss, masked_correct, masked_auprc) if aggregate=False
        """
        if success_logits is None or target_progress is None:
            # Return zero tensors instead of floats to maintain tensor consistency
            device = (
                success_logits.device
                if success_logits is not None
                else (target_progress.device if target_progress is not None else torch.device("cpu"))
            )
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Normalize inputs to tensors
        if isinstance(success_logits, list):
            success_logits = torch.stack(success_logits)
        if isinstance(target_progress, list):
            target_progress = torch.stack(target_progress)

        # Get base thresholds from config
        min_success = self.config.data.min_success
        max_success_default = self.config.data.max_success

        # Compute per-sample max_success thresholds if data_source is available
        if data_source is not None:
            max_success_tensor = torch.tensor(
                [
                    self.dataset_success_percent.get(ds if isinstance(ds, str) else str(ds), max_success_default)
                    for ds in data_source
                ],
                dtype=target_progress.dtype,
                device=target_progress.device,
            )
        else:
            max_success_tensor = torch.full(
                (success_logits.shape[0],),
                max_success_default,
                dtype=target_progress.dtype,
                device=target_progress.device,
            )

        # Handle Qwen downsampling: take every 2nd frame if using Qwen and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        # Ensure success_logits matches target_progress length after downsampling
        if "Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image:
            target_progress = target_progress[:, ::2]

        # Generate success labels and mask vectorized
        # combined_mask: 1.0 where we should compute loss (low or high progress), 0.0 otherwise
        # success_labels: 0.0 for failure, 1.0 for success
        combined_mask = ((target_progress < min_success) | (target_progress > max_success_tensor.unsqueeze(1))).float()
        success_labels = (target_progress > max_success_tensor.unsqueeze(1)).float()

        if progress_loss_mask is not None:
            progress_loss_mask_t = progress_loss_mask.to(device=combined_mask.device, dtype=combined_mask.dtype)
            # Expand mask from (batch_size,) to (batch_size, seq_len)
            combined_mask = combined_mask * progress_loss_mask_t.unsqueeze(1)

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

        if aggregate:
            success_loss = masked_loss.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            mean_accuracy = masked_correct.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            success_loss = success_loss.mean()
            mean_accuracy = mean_accuracy.mean()

            # Compute AUPRC across all valid frames
            if success_probs_flat.numel() > 0 and len(torch.unique(success_labels_flat)) > 1:
                auprc = average_precision_score(
                    success_labels_flat.float().detach().cpu().numpy(),
                    success_probs_flat.float().detach().cpu().numpy(),
                )
                mean_auprc = torch.tensor(auprc, device=success_loss.device, dtype=torch.float32)
            else:
                mean_auprc = torch.tensor(0.0, device=success_loss.device, dtype=torch.float32)

            # Don't delete tensors that might be part of autograd graph
            # They will be cleaned up automatically after backward pass
            return success_loss, mean_accuracy, mean_auprc

        # For non-aggregate case, compute AUPRC per sample
        # Compute AUPRC per sample (per sequence)
        batch_size = success_logits.shape[0]
        masked_auprc = torch.zeros(batch_size, device=success_logits.device, dtype=torch.float32)

        for i in range(batch_size):
            sample_mask = combined_mask[i] > 0
            if sample_mask.sum() > 0:
                sample_probs = success_probs[i][sample_mask]
                sample_labels = success_labels[i][sample_mask]
                if len(torch.unique(sample_labels)) > 1:
                    auprc = average_precision_score(
                        sample_labels.detach().cpu().float().numpy(), sample_probs.detach().cpu().float().numpy()
                    )
                    masked_auprc[i] = auprc

        # Don't delete intermediate tensors as they may be needed for backward pass
        return masked_loss, masked_correct, masked_auprc

    def _compute_progress_loss_helper(
        self,
        progress_pred,
        target_progress,
        mask,
        aggregate: bool = False,
    ):
        """
        Helper function to compute progress loss.

        Args:
            progress_pred: Progress prediction tensors (can be tensor or list of tensors) of shape (batch_size, seq_len)
            target_progress: Target progress tensors (can be tensor or list of tensors) of shape (batch_size, seq_len)
            mask: Per-sample mask tensor of shape (batch_size,) with 1.0 for samples where we should compute loss
            aggregate: Whether to return the mean of the losses and correlations

        Returns:
            tuple: (progress_loss, spearman_correlation)
        """
        if progress_pred is None or target_progress is None:
            # Return zero tensors instead of floats to maintain tensor consistency
            device = (
                progress_pred.device
                if progress_pred is not None
                else (target_progress.device if target_progress is not None else torch.device("cpu"))
            )
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Normalize inputs to tensors
        if isinstance(progress_pred, list):
            progress_pred = torch.stack(progress_pred)
        if isinstance(target_progress, list):
            target_progress = torch.stack(target_progress)

        # Handle Qwen downsampling: take every 2nd frame if using Qwen and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        if "Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image:
            target_progress = target_progress[:, ::2]

        # Apply all masks together at once
        combined_mask = torch.ones_like(target_progress, dtype=torch.float32)
        mask_t = None
        if mask is not None:
            mask_t = mask.to(device=combined_mask.device, dtype=combined_mask.dtype)
            # Expand mask from (batch_size,) to (batch_size, seq_len)
            combined_mask = combined_mask * mask_t.unsqueeze(1)

        # If predict_last_frame_progress is True, only compute loss for the last frame
        last_frame_mask = None
        if self.config.loss.predict_last_frame_progress:
            # Create a mask that only selects the last frame for each sequence
            last_frame_mask = torch.zeros_like(combined_mask, dtype=torch.float32)
            last_frame_mask[:, -1] = 1.0  # Set last frame to 1.0 for all sequences
            combined_mask = combined_mask * last_frame_mask

        # Compute MSE loss per frame
        loss_per_frame = F.mse_loss(progress_pred.float(), target_progress.float(), reduction="none")
        masked_loss = loss_per_frame * combined_mask
        spearman_correlations = compute_spearman_correlation(
            progress_pred, target_progress, aggregate=False, mask=combined_mask
        )

        if aggregate:
            # Average per sample, then take mean across batch
            progress_losses = masked_loss.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            progress_loss = progress_losses.mean()
            mean_spearman = spearman_correlations.mean()
            # Don't delete tensors that might be part of autograd graph
            # They will be cleaned up automatically after backward pass
            return progress_loss, mean_spearman

        # For non-aggregate case, return the tensors as-is
        # Don't delete intermediate tensors as they may be needed for backward pass
        return masked_loss, spearman_correlations

    def forward_model(self, model, inputs, sample_type="progress"):
        """Forward pass for the model."""
        with _timer("time/forward", timing_raw=self.timing_raw):
            if "rewind" in self.config.model.base_model_id:
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

            self.timing_raw.update(model_timing_raw)
            return model_output, model_timing_raw

    def _compute_progress_loss(self, model, inputs, return_outputs=False, training=True):
        """Compute progress prediction loss."""
        model_output, _ = self.forward_model(model, inputs, sample_type="progress")
        progress_logits = model_output.progress_logits
        progress_pred = progress_logits["A"]
        progress_target = inputs["target_progress"]
        progress_target_mask = inputs["target_progress_mask"]

        # [B, T], [B]
        progress_loss_all, spearman_corr_all = self._compute_progress_loss_helper(
            progress_pred,
            progress_target,
            progress_target_mask,
            aggregate=False,
        )

        # Handle Qwen downsampling: take every 2nd frame if using Qwen and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        # This downsampling is needed for creating the combined_mask that matches progress_loss_all shape
        # progress_target should already have requires_grad=False (set in collator), but clone to avoid view issues
        if "Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image:
            # Create downsampled version - use contiguous() to ensure it's not a view that breaks DDP
            progress_target_for_mask = progress_target[:, ::2].contiguous()
        else:
            progress_target_for_mask = progress_target

        # Apply all masks together at once
        # Create combined_mask without requires_grad to ensure it's not part of computation graph
        combined_mask = torch.ones_like(progress_target_for_mask, dtype=torch.float32, requires_grad=False)
        if progress_target_mask is not None:
            # progress_target_mask should already have requires_grad=False (set in collator)
            mask_t = progress_target_mask.to(device=combined_mask.device, dtype=combined_mask.dtype)
            # Expand mask from (batch_size,) to (batch_size, seq_len)
            combined_mask = combined_mask * mask_t.unsqueeze(1)

        progress_loss_all = progress_loss_all * combined_mask
        progress_loss = progress_loss_all.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
        progress_loss = progress_loss.mean()
        final_loss = progress_loss

        if self.config.model.train_success_head:
            success_logits = model_output.success_logits
            success_pred = success_logits["A"]
            data_source = inputs["data_source"]
            # Use the same downsampled target for success loss
            progress_target_for_success = (
                progress_target_for_mask
                if ("Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image)
                else progress_target
            )
            success_loss_all, success_acc_all, success_auprc_all = self._compute_success_loss_helper(
                success_pred,
                progress_target_for_success,
                progress_loss_mask=progress_target_mask,
                data_source=data_source,
                aggregate=False,
            )
            success_loss = success_loss_all.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            success_accuracy = success_acc_all.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            success_auprc = success_auprc_all.mean()
            success_loss = success_loss.mean()
            success_accuracy = success_accuracy.mean()
            final_loss = progress_loss + success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            rank_0_print(f"Warning: NaN detected in progress loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"

            # split spearman by data gen strategy
            strats = set(inputs["data_gen_strategy"])
            for strat in strats:
                mask = [1 if s == strat else 0 for s in inputs["data_gen_strategy"]]
                mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                masked_spearman = spearman_corr_all[mask == 1].detach() if training else spearman_corr_all[mask == 1]
                masked_loss = progress_loss_all[mask == 1].detach() if training else progress_loss_all[mask == 1]
                outputs_dict.update({
                    f"{prefix}_strat_spearman_corr/{strat}": masked_spearman.mean().item(),
                    f"{prefix}_strat_prog_loss/{strat}": masked_loss.mean().item(),
                })

            # split spearman by data source
            data_sources = set(inputs["data_source"])
            for data_source in data_sources:
                mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                masked_spearman = spearman_corr_all[mask == 1].detach() if training else spearman_corr_all[mask == 1]
                masked_loss = progress_loss_all[mask == 1].detach() if training else progress_loss_all[mask == 1]
                outputs_dict.update({
                    f"{prefix}_ds_spearman_corr/{data_source}": masked_spearman.mean().item(),
                    f"{prefix}_ds_prog_loss/{data_source}": masked_loss.mean().item(),
                })

            # Compute average Spearman correlation across trajectories A and B
            spearman_values = []
            if isinstance(spearman_corr_all, torch.Tensor):
                spearman_values.append(spearman_corr_all.mean().item())
            else:
                spearman_values.append(spearman_corr_all)

            avg_spearman = np.mean(spearman_values) if spearman_values else 0.0

            outputs_dict.update({
                f"{prefix}/prog_loss": progress_loss.item(),
                f"{prefix}/spearman_corr_avg": avg_spearman,
            })

            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/success_loss": success_loss.item(),
                    f"{prefix}/success_accuracy": success_accuracy.item(),
                    f"{prefix}/success_auprc": success_auprc.item(),
                })

            # DO NOT delete any tensors during training - they are part of the autograd graph
            # and must remain until backward completes. DDP will fail if we delete them prematurely.
            # Even during evaluation, be cautious - only delete if absolutely necessary and
            # after confirming they're not part of any computation graph.

        # Don't delete tensors that are part of the computation graph
        # They will be cleaned up automatically after backward pass completes
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
        target_progress_A_mask = inputs["target_progress_A_mask"]
        target_progress_A = inputs["target_progress_A"]
        data_source = inputs["data_source"]

        if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
            progress_pred_A = progress_logits["A"]

            progress_loss, spearman_corr = self._compute_progress_loss_helper(
                progress_pred_A,
                target_progress_A,
                mask=target_progress_A_mask,
                aggregate=True,
            )
            final_loss = preference_loss + progress_loss

        if self.config.model.train_success_head:
            success_logits = model_outputs.success_logits
            success_logits = success_logits["A"]

            success_loss, success_accuracy, success_auprc = self._compute_success_loss_helper(
                success_logits,
                target_progress_A,
                progress_loss_mask=target_progress_A_mask,
                data_source=data_source,
                aggregate=True,
            )
            final_loss = final_loss + success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            rank_0_print(f"Warning: NaN detected in preference loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"
            rejected_data_gen_strategy = inputs["rejected_data_gen_strategy"]

            if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
                outputs_dict.update({
                    f"{prefix}/pref_progress_loss": progress_loss.item(),
                    f"{prefix}/pref_spearman_corr": spearman_corr.item(),
                })

            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/pref_success_loss": success_loss.item(),
                    f"{prefix}/pref_success_accuracy": success_accuracy.item(),
                    f"{prefix}/pref_success_auprc": success_auprc.item(),
                })

            if preference_loss is not None:
                # Compute preference accuracy for training monitoring
                preference_probs = torch.sigmoid(preference_scores)
                preference_predictions = (preference_probs > 0.5).float()
                preference_accuracy = (preference_predictions == preference_labels).float()

                # split acc by data gen strategy
                rejected_strats = set(rejected_data_gen_strategy)
                for strat in rejected_strats:
                    mask = [1 if s == strat else 0 for s in rejected_data_gen_strategy]
                    mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                    # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                    masked_acc = preference_accuracy[mask == 1].detach() if training else preference_accuracy[mask == 1]
                    masked_loss = (
                        preference_loss_all[mask == 1].detach() if training else preference_loss_all[mask == 1]
                    )
                    outputs_dict.update({
                        f"{prefix}_strat_pref_acc/{strat}": masked_acc.mean().item(),
                        f"{prefix}_strat_pref_loss/{strat}": masked_loss.mean().item(),
                    })

                # split acc by data source
                data_sources = set(inputs["data_source"])
                for data_source in data_sources:
                    mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                    mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                    # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                    masked_acc = preference_accuracy[mask == 1].detach() if training else preference_accuracy[mask == 1]
                    masked_loss = (
                        preference_loss_all[mask == 1].detach() if training else preference_loss_all[mask == 1]
                    )
                    outputs_dict.update({
                        f"{prefix}_ds/pref_acc_{data_source}": masked_acc.mean().item(),
                        f"{prefix}_ds/pref_loss_{data_source}": masked_loss.mean().item(),
                    })

                outputs_dict.update({
                    # "preference_scores": preference_scores,
                    # "preference_labels": preference_labels,
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
        # Single forward pass with batched inputs (already batched by collator)
        # Batch structure: [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        batched_outputs, _ = self.forward_model(model, inputs, sample_type="similarity")

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

        # Handle progress_logits
        progress_logits_ref_sim = None
        progress_logits_ref_diff = None
        if batched_outputs.progress_logits is not None and batched_outputs.progress_logits.get("A") is not None:
            progress_A = batched_outputs.progress_logits["A"]
            if isinstance(progress_A, list):
                # List format - split by even/odd indices
                progress_logits_ref_sim = {"A": progress_A[::2], "B": None}
                progress_logits_ref_diff = {"A": progress_A[1::2], "B": None}
            elif isinstance(progress_A, torch.Tensor):
                # Tensor format - split along batch dimension
                progress_logits_ref_sim = {"A": progress_A[::2], "B": None}
                progress_logits_ref_diff = {"A": progress_A[1::2], "B": None}

        # Handle success_logits
        success_logits_ref_sim = None
        success_logits_ref_diff = None
        if batched_outputs.success_logits is not None and batched_outputs.success_logits.get("A") is not None:
            success_A = batched_outputs.success_logits["A"]
            if isinstance(success_A, list):
                # List format - split by even/odd indices
                success_logits_ref_sim = {"A": success_A[::2], "B": None}
                success_logits_ref_diff = {"A": success_A[1::2], "B": None}
            elif isinstance(success_A, torch.Tensor):
                # Tensor format - split along batch dimension
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
        # Compute progress and success loss for the reference trajectory
        # =========================================================================================
        target_progress_ref_mask = inputs["target_progress_ref_mask"]
        target_progress_ref = inputs["target_progress_ref"]
        data_source = inputs["data_source"]

        if self.config.model.train_progress_head and self.config.training.predict_sim_progress:
            progress_logits_ref_sim = model_outputs_ref_sim.progress_logits
            progress_pred_ref_sim_A = progress_logits_ref_sim["A"]

            progress_loss, spearman_corr = self._compute_progress_loss_helper(
                progress_pred_ref_sim_A,
                target_progress_ref,
                mask=target_progress_ref_mask,
                aggregate=True,
            )
            final_loss = similarity_loss + progress_loss

        if self.config.model.train_success_head:
            success_logits_ref_sim = model_outputs_ref_sim.success_logits
            success_logits_ref_sim_A = success_logits_ref_sim["A"]

            success_loss, success_accuracy, success_auprc = self._compute_success_loss_helper(
                success_logits_ref_sim_A,
                target_progress_ref,
                progress_loss_mask=target_progress_ref_mask,
                data_source=data_source,
                aggregate=True,
            )
            final_loss = final_loss + success_loss

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            rank_0_print(f"Warning: NaN detected in similarity loss, replacing with 0.0")
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

            # Split metrics by data generation strategy
            if data_gen_strategy is not None:
                strats = set(data_gen_strategy)
                for strat in strats:
                    mask = [1 if s == strat else 0 for s in data_gen_strategy]
                    mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                    # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                    masked_acc = similarity_correct[mask == 1].detach() if training else similarity_correct[mask == 1]
                    masked_loss = (
                        similarity_loss_all[mask == 1].detach() if training else similarity_loss_all[mask == 1]
                    )
                    masked_margin = similarity_margin[mask == 1] if not training else similarity_margin[mask == 1]
                    outputs_dict.update({
                        f"{prefix}_strat_sim_acc/{strat}": masked_acc.mean().item(),
                        f"{prefix}_strat_sim_loss/{strat}": masked_loss.mean().item(),
                        f"{prefix}_strat_sim_margin/{strat}": masked_margin.mean().item()
                        if masked_margin.numel() > 0
                        else 0.0,
                    })

            # Split metrics by data source
            data_sources = set(inputs["data_source"])
            for data_source in data_sources:
                mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                masked_acc = similarity_correct[mask == 1].detach() if training else similarity_correct[mask == 1]
                masked_loss = similarity_loss_all[mask == 1].detach() if training else similarity_loss_all[mask == 1]
                masked_margin = similarity_margin[mask == 1]
                outputs_dict.update({
                    f"{prefix}_ds_sim_acc/{data_source}": masked_acc.mean().item(),
                    f"{prefix}_ds_sim_loss/{data_source}": masked_loss.mean().item(),
                    f"{prefix}_ds_sim_margin/{data_source}": masked_margin.mean().item()
                    if masked_margin.numel() > 0
                    else 0.0,
                })

            # Add main metrics
            outputs_dict.update({
                f"{prefix}/similarity_loss": similarity_loss.item(),
                f"{prefix}/similarity_accuracy": similarity_accuracy.item(),
                f"{prefix}/similarity_margin": avg_similarity_margin.item(),
            })

            # Add progress loss metrics if computed
            if self.config.model.train_progress_head and self.config.training.predict_sim_progress:
                outputs_dict.update({
                    f"{prefix}/sim_progress_loss": progress_loss.item(),
                    f"{prefix}/sim_spearman_corr_avg": spearman_corr.item(),
                })

            # Add success loss metrics if computed
            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/sim_success_loss": success_loss.item(),
                    f"{prefix}/sim_success_accuracy": success_accuracy.item(),
                    f"{prefix}/sim_success_auprc": success_auprc.item(),
                })

            return final_loss, outputs_dict

        return final_loss
