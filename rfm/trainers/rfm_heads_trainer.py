import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Trainer
import matplotlib.pyplot as plt
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

        # Log to console on rank 0
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
            "success_failure": "succ_fail",
            "wrong_task": "wrong_task",
        }

        rank_0_print(f"\n\n\n\n")
        rank_0_print(f"=" * 100)
        rank_0_print(f"Running custom evaluations: {eval_types}")
        rank_0_print(f"=" * 100)

        for eval_type in eval_types:
            rank_0_print(f"Running evaluation for: {eval_type}")

            datasets = getattr(self.config.custom_eval, eval_type)
            eval_datasets_name = datasets

            for eval_dataset in eval_datasets_name:
                eval_cfg = copy.deepcopy(self.config.data)
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

                for batch in tqdm(
                    dataloader,
                    desc=f"Running {eval_type}, ds: {eval_dataset}, batch size: {self.config.training.per_device_eval_batch_size}",
                ):
                    batch = self._prepare_inputs(batch)

                    if eval_type in ["reward_alignment", "policy_ranking", "confusion_matrix"]:
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

                        # Gather everything
                        progress_pred = self.accelerator.gather_for_metrics(progress_pred)
                        target_progress = self.accelerator.gather_for_metrics(progress_samples["target_progress"])

                        # Gather metadata fields
                        metadata_fields = ["task", "data_source", "data_gen_strategy", "quality_labels", "metadata"]
                        gathered_metadata_dict = {}

                        if dist.is_initialized():
                            world_size = dist.get_world_size()
                            for field in metadata_fields:
                                gathered = [None] * world_size
                                dist.all_gather_object(gathered, progress_samples[field])
                                gathered_metadata_dict[field] = [item for sublist in gathered for item in sublist]
                                # Clean up gathered list immediately
                                del gathered
                        else:
                            for field in metadata_fields:
                                gathered_metadata_dict[field] = progress_samples[field]

                        # Handle success predictions if needed
                        success_pred_gathered = None
                        if self.config.model.train_success_head:
                            success_pred = outputs.success_logits["A"]
                            if isinstance(success_pred, list):
                                success_pred = (
                                    torch.stack(success_pred)
                                    if isinstance(success_pred[0], torch.Tensor)
                                    else torch.tensor(success_pred, device=self.accelerator.device)
                                )
                            success_binary = (torch.sigmoid(success_pred) > 0.5).float()
                            success_pred_gathered = self.accelerator.gather_for_metrics(success_binary)
                            # Clean up intermediate tensors
                            del success_pred, success_binary

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

                    elif eval_type == "success_failure":
                        # TODO: implement success failure evaluation on VQA
                        preference_samples = batch["preference_inputs"]
                        with torch.no_grad():
                            outputs, _ = self.forward_model(self.model, preference_samples, sample_type="preference")
                        pref_logits = outputs.pref_logits

                        # Gather predictions and labels across all ranks
                        pref_logits = self.accelerator.gather_for_metrics(pref_logits)
                        preference_labels = self.accelerator.gather_for_metrics(preference_samples["preference_labels"])

                        # Gather non-tensor metadata using all_gather_object
                        if dist.is_initialized():
                            world_size = dist.get_world_size()

                            # Gather each metadata field separately
                            gathered_task = [None] * world_size
                            gathered_data_source = [None] * world_size
                            gathered_chosen_data_gen_strategy = [None] * world_size
                            gathered_rejected_data_gen_strategy = [None] * world_size
                            gathered_metadata = [None] * world_size

                            dist.all_gather_object(gathered_task, preference_samples["task"])
                            dist.all_gather_object(gathered_data_source, preference_samples["data_source"])
                            dist.all_gather_object(
                                gathered_chosen_data_gen_strategy, preference_samples["chosen_data_gen_strategy"]
                            )
                            dist.all_gather_object(
                                gathered_rejected_data_gen_strategy, preference_samples["rejected_data_gen_strategy"]
                            )
                            dist.all_gather_object(gathered_metadata, preference_samples["metadata"])

                            # Flatten gathered lists (each element is a list from one rank)
                            gathered_task = [item for sublist in gathered_task for item in sublist]
                            gathered_data_source = [item for sublist in gathered_data_source for item in sublist]
                            gathered_chosen_data_gen_strategy = [
                                item for sublist in gathered_chosen_data_gen_strategy for item in sublist
                            ]
                            gathered_rejected_data_gen_strategy = [
                                item for sublist in gathered_rejected_data_gen_strategy for item in sublist
                            ]
                            gathered_metadata = [item for sublist in gathered_metadata for item in sublist]
                        else:
                            # Single process - no gathering needed
                            gathered_task = preference_samples["task"]
                            gathered_data_source = preference_samples["data_source"]
                            gathered_chosen_data_gen_strategy = preference_samples["chosen_data_gen_strategy"]
                            gathered_rejected_data_gen_strategy = preference_samples["rejected_data_gen_strategy"]
                            gathered_metadata = preference_samples["metadata"]

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
                    # create task groups from eval_results
                    eval_metrics, task_groups, task_details = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                elif eval_type == "confusion_matrix":
                    confusion_plot, confusion_matrix = compute_eval_metrics(
                        eval_type, eval_results, self.config.data.progress_pred_type
                    )
                else:
                    raise ValueError(f"Unsupported eval type: {eval_type}")

                # Store metrics for all processes
                ds_name = DS_SHORT_NAME_MAPPING.get(eval_dataset, eval_dataset)
                metrics[ds_name][eval_type] = eval_metrics

                # Only log and visualize on main process
                if self.accelerator.is_main_process:
                    rank_0_print(f"Completed {eval_type} evaluation: {len(eval_results)} samples")
                    rank_0_print(f"Metrics: {metrics[ds_name][eval_type]}")
                    rank_0_print("=" * 50)

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
                        data = []
                        for task, group in task_groups.items():
                            quality_to_rews = collections.defaultdict(list)
                            for t in group:
                                quality_to_rews[t["quality_label"]].append(round(t["final_reward"], 2))
                            quality_to_rews = ",".join([f"{q}:{r}" for q, r in quality_to_rews.items()])
                            data.append([task, quality_to_rews])

                        columns = ["task", "quality_and_rews"]
                        self.logger.log_table(
                            f"{ds_name}/policy_ranking_samples",
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

        rank_0_print("=" * 50)
        rank_0_print("Finished running custom evaluations!")
        rank_0_print("=" * 50)

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
                rank_0_print("\n=== Custom RFM Evaluation Results (Aggregated) ===")
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
            tuple: (success_loss, success_accuracy)
        """
        if success_logits is None or target_progress is None:
            # Return zero tensors instead of floats to maintain tensor consistency
            device = (
                success_logits.device
                if success_logits is not None
                else (target_progress.device if target_progress is not None else torch.device("cpu"))
            )
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

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

        if aggregate:
            success_loss = masked_loss.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            mean_accuracy = masked_correct.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            success_loss = success_loss.mean()
            mean_accuracy = mean_accuracy.mean()
            # Don't delete tensors that might be part of autograd graph
            # They will be cleaned up automatically after backward pass
            return success_loss, mean_accuracy

        # For non-aggregate case, return the tensors as-is
        # Don't delete intermediate tensors as they may be needed for backward pass
        return masked_loss, masked_correct

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
            progress_target_for_success = progress_target_for_mask if ("Qwen" in self.config.model.base_model_id and not self.config.data.use_multi_image) else progress_target
            success_loss_all, success_acc_all = self._compute_success_loss_helper(
                success_pred,
                progress_target_for_success,
                progress_loss_mask=progress_target_mask,
                data_source=data_source,
                aggregate=False,
            )
            success_loss = success_loss_all.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
            success_accuracy = success_acc_all.sum(dim=1) / (combined_mask.sum(dim=1) + 1e-8)
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

            success_loss, success_accuracy = self._compute_success_loss_helper(
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
                    masked_loss = preference_loss_all[mask == 1].detach() if training else preference_loss_all[mask == 1]
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
                    masked_loss = preference_loss_all[mask == 1].detach() if training else preference_loss_all[mask == 1]
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
        """Compute similarity scoring loss (DPO-style)."""
        if "Qwen" in self.config.model.base_model_id or "SmolVLM" in self.config.model.base_model_id:
            ref_sim_inputs = {
                "input_ids": inputs["input_ids_ref_sim"],
                "attention_mask": inputs["attention_mask_ref_sim"],
                "pixel_values": inputs.get("pixel_values_ref_sim"),
                "pixel_values_videos": inputs.get("pixel_values_videos_ref_sim"),
                "image_grid_thw": inputs.get("image_grid_thw_ref_sim"),
                "video_grid_thw": inputs.get("video_grid_thw_ref_sim"),
            }

            # Prepare inputs for reference vs trajectory diff forward pass
            ref_diff_inputs = {
                "input_ids": inputs["input_ids_ref_diff"],
                "attention_mask": inputs["attention_mask_ref_diff"],
                "pixel_values": inputs.get("pixel_values_ref_diff"),
                "pixel_values_videos": inputs.get("pixel_values_videos_ref_diff"),
                "image_grid_thw": inputs.get("image_grid_thw_ref_diff"),
                "video_grid_thw": inputs.get("video_grid_thw_ref_diff"),
            }
        elif "rewind" in self.config.model.base_model_id:
            # use embeddings
            if self.config.data.load_embeddings:
                ref_sim_inputs = {
                    "video_embeddings": inputs["video_embeddings_ref_sim"],
                    "text_embeddings": inputs["text_embeddings_ref_sim"],
                }
                ref_diff_inputs = {
                    "video_embeddings": inputs["video_embeddings_ref_diff"],
                    "text_embeddings": inputs["text_embeddings_ref_diff"],
                }

        # Forward pass for reference vs trajectory sim
        model_outputs_ref_sim, _ = self.forward_model(model, ref_sim_inputs, sample_type="similarity")

        # Forward pass for reference vs trajectory diff
        model_outputs_ref_diff, _ = self.forward_model(model, ref_diff_inputs, sample_type="similarity")

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

            success_loss, success_accuracy = self._compute_success_loss_helper(
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

            data_gen_strategy = inputs["data_gen_strategy"]

            # Split metrics by data generation strategy
            if data_gen_strategy is not None:
                strats = set(data_gen_strategy)
                for strat in strats:
                    mask = [1 if s == strat else 0 for s in data_gen_strategy]
                    mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                    # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                    masked_acc = similarity_correct[mask == 1].detach() if training else similarity_correct[mask == 1]
                    masked_loss = similarity_loss_all[mask == 1].detach() if training else similarity_loss_all[mask == 1]
                    outputs_dict.update({
                        f"{prefix}_strat_sim_acc/{strat}": masked_acc.mean().item(),
                        f"{prefix}_strat_sim_loss/{strat}": masked_loss.mean().item(),
                    })

            # Split metrics by data source
            data_sources = set(inputs["data_source"])
            for data_source in data_sources:
                mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                mask = torch.tensor(mask, device=self.accelerator.device, requires_grad=False)
                # Detach indexed tensors to avoid DDP hook issues with boolean indexing
                masked_acc = similarity_correct[mask == 1].detach() if training else similarity_correct[mask == 1]
                masked_loss = similarity_loss_all[mask == 1].detach() if training else similarity_loss_all[mask == 1]
                outputs_dict.update({
                    f"{prefix}_ds_sim_acc/{data_source}": masked_acc.mean().item(),
                    f"{prefix}_ds_sim_loss/{data_source}": masked_loss.mean().item(),
                })

            # Add main metrics
            outputs_dict.update({
                f"{prefix}/similarity_loss": similarity_loss.item(),
                f"{prefix}/similarity_accuracy": similarity_accuracy.item(),
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
                })

            return final_loss, outputs_dict

        return final_loss
