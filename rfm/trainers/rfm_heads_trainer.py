import collections
import ast
from re import M, S
import wandb
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import List, Dict, Optional, Union, Any
import numpy as np
from tqdm import tqdm

from rfm.utils.logging import is_rank_0, rank_0_print
from rfm.utils.metrics import compute_spearman_correlation
from rfm.utils.logging import _timer


def reduce_metrics_with_accelerate(metrics: dict, accelerator):
    """
    Reduce multiple scalar metrics using Accelerate's built-in methods.
    Handles cases where different processes have different metric keys.
    metrics: dict of {name: float or tensor}
    Returns dict with averaged metrics across all ranks.
    """
    if not metrics:
        return metrics

    try:
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
        
        all_unique_keys = sorted(list(all_unique_keys))
        
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
                reduced_val = accelerator.reduce(tensor_val, reduction="sum")
                
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
        
        # Step 5: Only return metrics that were originally present in this process
        final_metrics = {}
        for key in metrics.keys():
            final_metrics[key] = result_metrics[key]
        
        return final_metrics
        
    except Exception as e:
        # Fallback: return original metrics if reduction fails
        rank_0_print(f"Warning: reduce_metrics_with_accelerate failed with error: {e}. Returning original metrics.")
        fallback_metrics = {}
        for k, v in metrics.items():
            val = float(v) if not torch.is_tensor(v) else v.item()
            # Replace NaN with 0.0 in fallback
            fallback_metrics[k] = 0.0 if np.isnan(val) else val
        return fallback_metrics


class RFMHeadsTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_metadata = collections.defaultdict(float)
        self.global_metadata = collections.defaultdict(float)
        self.timing_raw = collections.defaultdict(float)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        self.timing_raw = {}

        # Initialize log_metadata
        self.log_metadata = {}

        with _timer("time/training_step", timing_raw=self.timing_raw):
            # Call the parent training_step to handle all the standard training logic
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        num_progress = inputs.get("num_progress", 0)

        if num_preferences > 0 and preference_inputs:
            # Count data generation strategies from the rejected trajectories
            rejected_data_gen_strategy = preference_inputs.get("rejected_data_gen_strategy", [])
            if isinstance(rejected_data_gen_strategy, list) and len(rejected_data_gen_strategy) > 0:
                # Normalize keys we care about
                strat_counts = {
                    "num_trajs_rewind": 0,
                    "num_trajs_same_task": 0,
                    "num_trajs_different_task": 0,
                }
                for s in rejected_data_gen_strategy:
                    if s == "rewind_same_task":
                        strat_counts["num_trajs_rewind"] += 1
                    elif s == "suboptimal_same_task":
                        strat_counts["num_trajs_same_task"] += 1
                    elif s == "different_task":
                        strat_counts["num_trajs_different_task"] += 1

                self.log_metadata.update(strat_counts)

            data_sources = preference_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        # Update global metadata for training
        # add to total batch size and sum across all processes
        self.global_metadata["total_samples"] += num_preferences + num_similarities + num_progress
        self.global_metadata["total_preferences"] += num_preferences
        self.global_metadata["total_similarities"] += num_similarities
        self.global_metadata["total_progress"] += num_progress

        # Log custom losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_metadata()

        return loss

    def _log_metadata(self):
        """Log custom RFM losses to wandb and console."""
        if not self.log_metadata:
            return

        # Use local metrics (no aggregation needed for individual GPU metrics)
        log_metadata = reduce_metrics_with_accelerate(self.log_metadata, self.accelerator)

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
            **self.timing_raw,
            **log_metadata,
        }

        # Log global metadata
        global_metadata = reduce_metrics_with_accelerate(self.global_metadata, self.accelerator)
        log_global = {f"counts/{key}": global_metadata[key] for key in global_metadata}
        log_data.update(log_global)
        # make sure values are floats so they are loggable into wandb reports
        log_data = {k: float(v) for k, v in log_data.items()}

        # Log to wandb if available and configured (only on rank 0)
        if self.args.report_to and "wandb" in self.args.report_to and is_rank_0():
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(log_data)
            except ImportError:
                rank_0_print("Warning: wandb not available for logging custom losses")

        # Log to console on rank 0
        if is_rank_0():
            rank_0_print(f"Step {self.state.global_step} Custom Losses (Aggregated):")
            rank_0_print("-" * 50)
            for key in log_global:
                rank_0_print(f"  {key}: {log_global[key]}")

            rounded_times = {k: round(v, 2) for k, v in self.timing_raw.items()}
            rank_0_print(f"Timing raw: {rounded_times}")

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> Dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
        # Get the evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Set model to eval mode
        self.model.eval()

        # Run evaluation
        outputs = []
        with _timer("time/evaluate", timing_raw=self.timing_raw):
            with torch.no_grad():
                for step, inputs in tqdm(
                    enumerate(eval_dataloader),
                    total=len(eval_dataloader),
                    desc="Evaluating",
                ):
                    # Move inputs to device
                    inputs = self._prepare_inputs(inputs)

                    _, loss_dicts = self.compute_loss(self.model, inputs, return_outputs=True, training=False)
                    outputs.append(loss_dicts)

        # Aggregate outputs
        metrics = {}

        # assume that we already called .item() on the outputs
        keys = list(outputs[0].keys())
        for key in keys:
            metrics[key] = [output[key] for output in outputs if key in output]
            metrics[key] = np.array(metrics[key]).mean()

        # Aggregate metrics across all processes using accelerator
        metrics = reduce_metrics_with_accelerate(metrics, self.accelerator)

        # Log metrics
        if is_rank_0():
            rank_0_print(f"\n=== Custom RFM Evaluation Results (Aggregated) ===")
            for key, value in metrics.items():
                rank_0_print(f"{key}: {value:.6f}")
            rank_0_print("=" * 50)

        # Also log to wandb if available and configured (only on rank 0)
        if self.args.report_to and "wandb" in self.args.report_to and is_rank_0():
            if wandb.run is not None:
                wandb.log(metrics)

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """Compute loss for separate preference and similarity batches."""

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)

        # Initialize loss components and metadata
        total_loss = 0.0
        log_metadata = {}

        # Compute preference loss if we have preference samples

        if num_preferences > 0 and preference_inputs:
            with _timer("time/compute_preference_loss", timing_raw=self.timing_raw):
                preference_loss, progress_loss, loss_dict = self._compute_preference_loss(
                    model, preference_inputs, return_outputs=True, training=training
                )
            if self.config.model.train_preference_head:
                total_loss += preference_loss
            if self.config.model.train_progress_head:
                total_loss += progress_loss

            log_metadata.update(loss_dict)

        # Compute similarity loss if we have similarity samples
        if num_similarities > 0 and similarity_inputs:
            with _timer("time/compute_similarity_loss", timing_raw=self.timing_raw):
                similarity_loss, progress_loss, loss_dict = self._compute_similarity_loss(
                    model, similarity_inputs, return_outputs=True, training=training
                )
            if self.config.model.train_similarity_head:
                total_loss += similarity_loss
            if self.config.model.train_progress_head:
                total_loss += progress_loss
            log_metadata.update(loss_dict)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {**log_metadata, "total_loss": total_loss.item()}
            return total_loss, extra_info

        return total_loss

    def _compute_progress_loss(
        self,
        progress_logits,
        target_progress,
        frame_shape,
        target_progress_mask,
        trajectory_name="trajectory",
        aggregate: bool = False,
    ):
        """
        Helper function to compute progress loss for a single trajectory with frame shape splicing.

        Args:
            progress_logits: Progress prediction tensors (can be tensor or list of tensors)
            target_progress: Target progress tensors (can be tensor or list of tensors)
            frame_shape: List of frame shapes for splicing
            trajectory_name: Name of trajectory for logging/debugging
            aggregate: Whether to return the mean of the losses and correlations

        Returns:
            tuple: (progress_loss, spearman_correlation)
        """
        # Handle case where inputs might be tensors or lists
        if progress_logits is None or target_progress is None:
            return 0.0, 0.0

        # Ensure we have the same number of samples
        assert len(progress_logits) == len(target_progress), (
            f"{trajectory_name}: Progress logits and target progress have different batch sizes"
        )

        # Splice both predicted and target logits based on frame shapes
        spliced_progress_logits = []
        spliced_target_progress = []

        for i, (pred, target, shape) in enumerate(zip(progress_logits, target_progress, frame_shape)):
            num_frames = shape[0] if len(shape) > 0 else 0
            if "Qwen" in self.config.model.base_model_id:
                spliced_target = target[:num_frames][::2]
            else:
                spliced_target = target[:num_frames]

            spliced_progress_logits.append(pred)
            spliced_target_progress.append(spliced_target)

        # Compute MSE loss per element and then stack into a tensor
        progress_losses = []
        spearman_correlations = []

        for i, (pred, target) in enumerate(zip(spliced_progress_logits, spliced_target_progress)):
            loss = F.mse_loss(pred, target)
            progress_losses.append(loss)

            # Compute Spearman correlation for this sample
            spearman_corr = compute_spearman_correlation(pred, target)
            spearman_correlations.append(spearman_corr)

        if progress_losses:
            if aggregate:
                progress_loss = torch.stack(progress_losses) * target_progress_mask
                if target_progress_mask.sum() > 0:
                    progress_loss = progress_loss.sum() / target_progress_mask.sum()
                else:
                    progress_loss = progress_loss.mean()

                # Average the Spearman correlations
                mean_spearman = torch.stack(spearman_correlations) * target_progress_mask
                if target_progress_mask.sum() > 0:
                    mean_spearman = mean_spearman.sum() / target_progress_mask.sum()
                else:
                    mean_spearman = mean_spearman.mean()

                return progress_loss, mean_spearman
            else:
                progress_losses = torch.stack(progress_losses) * target_progress_mask
                spearman_correlations = torch.stack(spearman_correlations) * target_progress_mask
                return progress_losses, spearman_correlations

        return 0.0, 0.0

    def _compute_preference_loss(self, model, inputs, return_outputs=False, training=True):
        """Compute preference prediction loss using Bradley-Terry model."""
        # Single forward pass with both trajectories concatenated
        # The model should handle the preference prediction at the end
        with _timer("time/pref_forward", timing_raw=self.timing_raw):
            model_outputs, progress_logits, model_timing_raw = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values", None),
                pixel_values_videos=inputs.get("pixel_values_videos", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                sample_type="preference",
                timing_raw=self.timing_raw,
            )
            self.timing_raw.update(model_timing_raw)

        chosen_data_gen_strategy = inputs.get("chosen_data_gen_strategy", None)
        rejected_data_gen_strategy = inputs.get("rejected_data_gen_strategy", None)

        preference_loss = 0.0
        progress_loss = 0.0

        # Get preference labels (1 if first trajectory is preferred, 0 if second trajectory is preferred)
        preference_labels = inputs["preference_labels"]

        if self.config.model.train_preference_head:
            # Get preference scores from the preference head
            preference_scores = model_outputs.logits.squeeze(-1)  # [batch_size]

            # Binary cross entropy loss for preference prediction
            preference_loss_all = F.binary_cross_entropy_with_logits(
                preference_scores, preference_labels.float(), reduction="none"
            )
            preference_loss = preference_loss_all.mean()

        if self.config.model.train_progress_head:
            # Get frame shapes for splicing target progress to match predicted lengths
            # Since the order is randomized, we need to use preference labels to determine which is which
            chosen_frames_shape = inputs.get("chosen_frames_shape", None)
            rejected_frames_shape = inputs.get("rejected_frames_shape", None)

            # Determine which frame shape corresponds to which trajectory based on preference labels
            # preference_labels: 1.0 = first trajectory preferred, 0.0 = second trajectory preferred
            # We need to map this to chosen vs rejected for progress loss calculation

            # For each sample, determine which trajectory (first or second) is the chosen one
            batch_size = len(preference_labels)
            chosen_traj_shapes = []
            rejected_traj_shapes = []
            chosen_traj_progress_pred = []
            rejected_traj_progress_pred = []
            chosen_traj_progress_target = []
            rejected_traj_progress_target = []
            chosen_traj_progress_target_mask = []
            rejected_traj_progress_target_mask = []

            for i in range(batch_size):
                # First trajectory is preferred (chosen)
                chosen_traj_shapes.append(chosen_frames_shape[i])
                rejected_traj_shapes.append(rejected_frames_shape[i])

                chosen_traj_progress_target.append(inputs["target_progress_chosen"][i])
                rejected_traj_progress_target.append(inputs["target_progress_rejected"][i])
                chosen_traj_progress_target_mask.append(inputs["target_progress_chosen_mask"][i])
                rejected_traj_progress_target_mask.append(inputs["target_progress_rejected_mask"][i])

                if preference_labels[i] == 1.0:
                    chosen_traj_progress_pred.append(progress_logits["A"][i])
                    rejected_traj_progress_pred.append(progress_logits["B"][i])
                else:
                    # Second trajectory is preferred
                    chosen_traj_progress_pred.append(progress_logits["B"][i])
                    rejected_traj_progress_pred.append(progress_logits["A"][i])

            # Convert to tensors for the helper function
            chosen_traj_shapes = torch.stack(chosen_traj_shapes)
            rejected_traj_shapes = torch.stack(rejected_traj_shapes)
            chosen_traj_progress_target_mask = torch.stack(chosen_traj_progress_target_mask)
            rejected_traj_progress_target_mask = torch.stack(rejected_traj_progress_target_mask)

            # Compute progress loss for both trajectories using the helper function
            # Now we know which shape corresponds to which trajectory based on preference labels
            if self.config.model.train_progress_head:
                progress_loss_chosen, spearman_corr_chosen = self._compute_progress_loss(
                    chosen_traj_progress_pred,
                    chosen_traj_progress_target,
                    chosen_traj_shapes,
                    chosen_traj_progress_target_mask,
                    "A",
                    aggregate=False,
                )
                progress_loss_rejected, spearman_corr_rejected = self._compute_progress_loss(
                    rejected_traj_progress_pred,
                    rejected_traj_progress_target,
                    rejected_traj_shapes,
                    rejected_traj_progress_target_mask,
                    "B",
                    aggregate=False,
                )

                # Combine progress losses
                progress_loss = progress_loss_chosen.mean() + progress_loss_rejected.mean()

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"

            if self.config.model.train_preference_head and preference_loss is not None:
                # Compute preference accuracy for training monitoring
                preference_probs = torch.sigmoid(preference_scores)
                preference_predictions = (preference_probs > 0.5).float()
                preference_accuracy = (preference_predictions == preference_labels).float()

                # split acc by data gen strategy
                rejected_strats = set(rejected_data_gen_strategy)
                for strat in rejected_strats:
                    mask = [1 if s == strat else 0 for s in rejected_data_gen_strategy]
                    mask = torch.tensor(mask, device=self.accelerator.device)

                    outputs_dict.update(
                        {
                            f"{prefix}_strat_pref_acc/{strat}": (preference_accuracy[mask == 1]).mean().item(),
                            f"{prefix}_strat_pref_loss/{strat}": (preference_loss_all[mask == 1]).mean().item(),
                        }
                    )

                # split acc by data source
                data_sources = set(inputs["data_source"])
                for data_source in data_sources:
                    mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                    mask = torch.tensor(mask, device=self.accelerator.device)
                    outputs_dict.update(
                        {
                            f"{prefix}_ds/pref_acc_{data_source}": (preference_accuracy[mask == 1]).mean().item(),
                            f"{prefix}_ds/pref_loss_{data_source}": (preference_loss_all[mask == 1]).mean().item(),
                        }
                    )

                outputs_dict.update(
                    {
                        # "preference_scores": preference_scores,
                        # "preference_labels": preference_labels,
                        f"{prefix}/preference_loss": preference_loss.item(),
                        f"{prefix}/preference_accuracy": preference_accuracy.mean().item(),
                    }
                )

            if self.config.model.train_progress_head:
                # split spearman by data gen strategy
                rejected_strats = set(rejected_data_gen_strategy)
                for strat in rejected_strats:
                    mask = [1 if s == strat else 0 for s in rejected_data_gen_strategy]
                    mask = torch.tensor(mask, device=self.accelerator.device)
                    outputs_dict.update(
                        {
                            f"{prefix}_strat_spearman_corr/{strat}": (spearman_corr_rejected[mask == 1]).mean().item(),
                            f"{prefix}_strat_prog_loss/{strat}": (progress_loss_rejected[mask == 1]).mean().item(),
                        }
                    )

                # split spearman by data source
                data_sources = set(inputs["data_source"])
                for data_source in data_sources:
                    mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
                    mask = torch.tensor(mask, device=self.accelerator.device)
                    outputs_dict.update(
                        {
                            f"{prefix}_ds_spearman_corr/{data_source}": (spearman_corr_rejected[mask == 1])
                            .mean()
                            .item(),
                            f"{prefix}_ds_prog_loss/{data_source}": (progress_loss_rejected[mask == 1]).mean().item(),
                        }
                    )

                # Compute average Spearman correlation across trajectories A and B
                spearman_values = []
                if isinstance(spearman_corr_chosen, torch.Tensor):
                    spearman_values.append(spearman_corr_chosen.mean().item())
                else:
                    spearman_values.append(spearman_corr_chosen)

                if isinstance(spearman_corr_rejected, torch.Tensor):
                    spearman_values.append(spearman_corr_rejected.mean().item())
                else:
                    spearman_values.append(spearman_corr_rejected)

                avg_spearman = np.mean(spearman_values) if spearman_values else 0.0

                outputs_dict.update(
                    {
                        f"{prefix}/prog_loss_chosen": progress_loss_chosen.mean().item(),
                        f"{prefix}/prog_loss_rejected": progress_loss_rejected.mean().item(),
                        f"{prefix}/progress_loss": progress_loss.item(),
                        f"{prefix}/spearman_corr_avg": avg_spearman,
                    }
                )
            return preference_loss, progress_loss, outputs_dict
        return preference_loss, progress_loss

    def _compute_similarity_loss(self, model, inputs, return_outputs=False):
        """Compute similarity scoring loss (DPO-style)."""
        # Forward pass for reference vs trajectory sim
        model_outputs_ref_sim, progress_logits_ref_sim = model(
            input_ids=inputs["input_ids_ref_sim"],
            attention_mask=inputs["attention_mask_ref_sim"],
            pixel_values=inputs.get("pixel_values_ref_sim"),
            pixel_values_videos=inputs.get("pixel_values_videos_ref_sim"),
            image_grid_thw=inputs.get("image_grid_thw_ref_sim"),
            video_grid_thw=inputs.get("video_grid_thw_ref_sim"),
            sample_type="similarity",
        )

        # Forward pass for reference vs trajectory diff
        model_outputs_ref_diff, progress_logits_ref_diff = model(
            input_ids=inputs["input_ids_ref_diff"],
            attention_mask=inputs["attention_mask_ref_diff"],
            pixel_values=inputs.get("pixel_values_ref_diff"),
            pixel_values_videos=inputs.get("pixel_values_videos_ref_diff"),
            image_grid_thw=inputs.get("image_grid_thw_ref_diff"),
            video_grid_thw=inputs.get("video_grid_thw_ref_diff"),
            sample_type="similarity",
        )

        # Extract similarity scores
        score_ref_sim = model_outputs_ref_sim.logits.squeeze(-1)
        score_ref_diff = model_outputs_ref_diff.logits.squeeze(-1)

        # Compute DPO-style loss: encourage trajectory A to be more similar to reference than trajectory B
        # This assumes trajectory A is the "better" trajectory (more similar to reference)
        similarity_loss = -F.logsigmoid(self.config.training.beta * (score_ref_sim - score_ref_diff)).mean()

        # Get frame shapes for splicing target progress to match predicted lengths
        # For similarity samples, we use traj_sim_frames_shape and traj_diff_frames_shape
        ref_frames_shape = inputs.get("ref_frames_shape", None)
        traj_sim_frames_shape = inputs.get("traj_sim_frames_shape", None)
        traj_diff_frames_shape = inputs.get("traj_diff_frames_shape", None)

        # Compute progress loss for both forward passes
        # Both forward pass compute for the reference trajectory
        # A is the reference trajectory and B is the trajectory to compare to
        progress_loss_ref, spearman_corr_ref = self._compute_progress_loss(
            progress_logits_ref_sim["A"],
            inputs["target_progress_ref"],
            ref_frames_shape,
            inputs["target_progress_ref_mask"],
            "A",
        )
        progress_loss_sim, spearman_corr_sim = self._compute_progress_loss(
            progress_logits_ref_sim["B"],
            inputs["target_progress_A"],
            traj_sim_frames_shape,
            inputs["target_progress_sim_mask"],
            "sim",
        )

        progress_loss_diff, spearman_corr_diff = self._compute_progress_loss(
            progress_logits_ref_diff["B"],
            inputs["target_progress_B"],
            traj_diff_frames_shape,
            inputs["target_progress_diff_mask"],
            "diff",
        )

        progress_loss = progress_loss_ref + progress_loss_sim + progress_loss_diff

        # Combine losses
        total_loss = similarity_loss + progress_loss_ref + progress_loss_sim + progress_loss_diff

        if return_outputs:
            # Compute average Spearman correlation across all trajectories (ref, sim, diff)
            spearman_values = []
            for corr in [spearman_corr_ref, spearman_corr_sim, spearman_corr_diff]:
                if isinstance(corr, torch.Tensor):
                    spearman_values.append(corr.item())
                else:
                    spearman_values.append(corr)

            avg_spearman = np.mean(spearman_values) if spearman_values else 0.0

            outputs_dict = {
                "score_ref_sim": score_ref_sim,
                "score_ref_diff": score_ref_diff,
                "similarity_loss": similarity_loss.item(),
                "progress_loss_ref": progress_loss_ref.item(),
                "progress_loss_sim": progress_loss_sim.item(),
                "progress_loss_diff": progress_loss_diff.item(),
                "progress_loss": progress_loss.item(),
                "predicted_progress_A": progress_logits_ref_sim["A"],
                "predicted_progress_B": progress_logits_ref_sim["B"],
                "target_progress_A": inputs["target_progress_A"],
                "target_progress_B": inputs["target_progress_B"],
                "spearman_corr_avg": avg_spearman,
            }
            return total_loss, outputs_dict
        return total_loss
