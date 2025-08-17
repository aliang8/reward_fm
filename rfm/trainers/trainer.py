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
from rfm.utils.metrics import compute_auc, compute_spearman_correlation


class RFMTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        # Initialize custom loss tracking
        self.custom_losses = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        # Call the parent training_step to handle all the standard training logic
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Log custom losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_custom_losses()

        return loss

    def _log_custom_losses(self):
        """Log custom RFM losses to wandb and console."""
        if not self.custom_losses:
            return

        # Aggregate custom losses across all processes if using distributed training
        aggregated_losses = self._aggregate_custom_losses()
        aggregated_losses = {f"train/{key}": aggregated_losses[key] for key in aggregated_losses}

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
        }
        log_data.update(aggregated_losses)

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
            log_keys = [
                "preference_loss",
                "preference_accuracy",
                "similarity_loss",
                "progress_loss",
                "spearman_corr_avg",
            ]
            for key in log_keys:
                if f"train/{key}" in aggregated_losses:
                    rank_0_print(f"  {key}: {aggregated_losses[f'train/{key}']:.6f}")

    def _aggregate_custom_losses(self):
        """Aggregate custom losses across all processes using all_reduce."""
        if not self.custom_losses:
            return {}

        import torch.distributed as dist

        # If not using distributed training, return losses as-is
        if not dist.is_initialized():
            return self.custom_losses.copy()

        aggregated = {}

        # Aggregate loss values (averages)
        loss_keys = ["preference_loss", "similarity_loss", "progress_loss", "preference_accuracy", "spearman_corr_avg"]
        for key in loss_keys:
            if key in self.custom_losses:
                # Convert to tensor for all_reduce
                loss_tensor = torch.tensor(self.custom_losses[key], device=self.accelerator.device)

                # Sum across all processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

                # Average by world size
                aggregated[key] = (loss_tensor / dist.get_world_size()).item()

        # Aggregate count values (sums)
        count_keys = ["preference_count", "similarity_count", "progress_count"]
        for key in count_keys:
            if key in self.custom_losses:
                # Convert to tensor for all_reduce
                count_tensor = torch.tensor(self.custom_losses[key], device=self.accelerator.device)

                # Sum across all processes
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

                # Keep as sum (total across all processes)
                aggregated[key] = count_tensor.item()

        return aggregated

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
        # Get the evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Set model to eval mode
        self.model.eval()

        # Run evaluation
        outputs = []
        with torch.no_grad():
            for step, inputs in tqdm(
                enumerate(eval_dataloader),
                total=len(eval_dataloader),
                desc="Evaluating",
            ):
                # Move inputs to device
                inputs = self._prepare_inputs(inputs)

                _, loss_dicts = self.compute_loss(self.model, inputs, return_outputs=True)
                outputs.append(loss_dicts)

        # Aggregate outputs
        log_keys = [
            "preference_loss",
            "similarity_loss",
            "progress_loss",
            "preference_accuracy",
            "spearman_corr_avg",
        ]  # TODO: add progress_loss_A, progress_loss_B, progress_loss_ref, progress_loss_sim, progress_loss_diff
        aggregated_outputs = {}

        # assume that we already called .item() on the outputs
        for key in log_keys:
            if key in outputs[0]:
                aggregated_outputs[key] = [output[key] for output in outputs if key in output]
                aggregated_outputs[key] = np.array(aggregated_outputs[key]).mean()

        # Compute metrics
        metrics = {f"eval/{key}": aggregated_outputs[key] for key in aggregated_outputs}

        # Log metrics
        if is_rank_0():
            rank_0_print(f"\n=== Custom RFM Evaluation Results ===")
            for key, value in metrics.items():
                rank_0_print(f"{key}: {value:.6f}")
            rank_0_print("=" * 50)

        # Also log to wandb if available and configured (only on rank 0)
        if self.args.report_to and "wandb" in self.args.report_to and is_rank_0():
            if wandb.run is not None:
                wandb.log(metrics)

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for separate preference and similarity batches."""

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)

        # Initialize loss components and metadata
        total_loss = 0.0
        loss_metadata = {}

        # Compute preference loss if we have preference samples
        if num_preferences > 0 and preference_inputs:
            preference_loss, progress_loss, loss_dict = self._compute_preference_loss(
                model, preference_inputs, return_outputs=True
            )
            if self.config.model.train_preference_head:
                total_loss += preference_loss
            if self.config.model.train_progress_head:
                total_loss += progress_loss

            loss_metadata.update(loss_dict)

        # Compute similarity loss if we have similarity samples
        if num_similarities > 0 and similarity_inputs:
            similarity_loss, progress_loss, loss_dict = self._compute_similarity_loss(
                model, similarity_inputs, return_outputs=True
            )
            if self.config.model.train_similarity_head:
                total_loss += similarity_loss
            if self.config.model.train_progress_head:
                total_loss += progress_loss
            loss_metadata.update(loss_dict)

        # Always store custom losses for logging (even when return_outputs=False)
        self.custom_losses = loss_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {
                **loss_metadata,
                "total_loss": total_loss,
                "batch_size": num_preferences + num_similarities,
            }
            return total_loss, extra_info

        return total_loss

    def _compute_progress_loss(
        self,
        progress_logits,
        target_progress,
        frame_shape,
        trajectory_name="trajectory",
    ):
        """
        Helper function to compute progress loss for a single trajectory with frame shape splicing.

        Args:
            progress_logits: Progress prediction tensors (can be tensor or list of tensors)
            target_progress: Target progress tensors (can be tensor or list of tensors)
            frame_shape: List of frame shapes for splicing
            trajectory_name: Name of trajectory for logging/debugging

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
            # Extract frame count from shape (first dimension)
            num_frames = shape[0] if len(shape) > 0 else 0
            num_frames = num_frames // 2  # add this because of temporal_patch_size
            # Splice both predicted and target to match frame count
            spliced_pred = pred[:num_frames]
            spliced_target = target[:num_frames]  # TODO: check if this is correct
            spliced_progress_logits.append(spliced_pred)
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
            progress_loss = torch.stack(progress_losses).mean()
            # Average the Spearman correlations
            mean_spearman = torch.stack(spearman_correlations).mean()
            return progress_loss, mean_spearman
        return 0.0, 0.0

    def _compute_preference_loss(self, model, inputs, return_outputs=False):
        """Compute preference prediction loss using Bradley-Terry model."""
        # Single forward pass with both trajectories concatenated
        # The model should handle the preference prediction at the end
        model_outputs, progress_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            sample_type="preference",
        )

        preference_loss = 0.0
        progress_loss = 0.0

        # Get preference labels (1 if first trajectory is preferred, 0 if second trajectory is preferred)
        preference_labels = inputs["preference_labels"]

        if self.config.model.train_preference_head:
            # Get preference scores from the preference head
            preference_scores = model_outputs.logits.squeeze(-1)  # [batch_size]

            # Binary cross entropy loss for preference prediction
            preference_loss = F.binary_cross_entropy_with_logits(preference_scores, preference_labels.float())

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
            first_trajectory_shapes = []
            second_trajectory_shapes = []

            for i in range(batch_size):
                if preference_labels[i] == 1.0:
                    # First trajectory is preferred (chosen)
                    first_trajectory_shapes.append(chosen_frames_shape[i])
                    second_trajectory_shapes.append(rejected_frames_shape[i])
                else:
                    # Second trajectory is preferred (chosen)
                    first_trajectory_shapes.append(rejected_frames_shape[i])
                    second_trajectory_shapes.append(chosen_frames_shape[i])

            # Convert to tensors for the helper function
            first_trajectory_shapes = torch.stack(first_trajectory_shapes)
            second_trajectory_shapes = torch.stack(second_trajectory_shapes)

            # Compute progress loss for both trajectories using the helper function
            # Now we know which shape corresponds to which trajectory based on preference labels
            if self.config.model.train_progress_head:
                progress_loss_A, spearman_corr_A = self._compute_progress_loss(
                    progress_logits["A"], inputs["target_progress_A"], first_trajectory_shapes, "A"
                )
                progress_loss_B, spearman_corr_B = self._compute_progress_loss(
                    progress_logits["B"],
                    inputs["target_progress_B"],
                    second_trajectory_shapes,
                    "B",
                )

                # Combine progress losses
                progress_loss = progress_loss_A + progress_loss_B

        if return_outputs:
            outputs_dict = {}
            if self.config.model.train_preference_head and preference_loss is not None:
                # Compute preference accuracy for training monitoring
                preference_probs = torch.sigmoid(preference_scores)
                preference_predictions = (preference_probs > 0.5).float()
                preference_accuracy = (preference_predictions == preference_labels).float().mean().item()
                outputs_dict.update(
                    {
                        "preference_scores": preference_scores,
                        "preference_labels": preference_labels,
                        "preference_loss": preference_loss.item(),
                        "preference_accuracy": preference_accuracy,
                    }
                )

            if self.config.model.train_progress_head:
                # Compute average Spearman correlation across trajectories A and B
                spearman_values = []
                if isinstance(spearman_corr_A, torch.Tensor):
                    spearman_values.append(spearman_corr_A.item())
                else:
                    spearman_values.append(spearman_corr_A)
                if isinstance(spearman_corr_B, torch.Tensor):
                    spearman_values.append(spearman_corr_B.item())
                else:
                    spearman_values.append(spearman_corr_B)

                avg_spearman = np.mean(spearman_values) if spearman_values else 0.0

                outputs_dict.update(
                    {
                        "progress_loss_A": progress_loss_A.item(),
                        "progress_loss_B": progress_loss_B.item(),
                        "progress_loss": progress_loss.item(),
                        "predicted_progress_A": progress_logits["A"],
                        "predicted_progress_B": progress_logits["B"],
                        "target_progress_A": inputs["target_progress_A"],
                        "target_progress_B": inputs["target_progress_B"],
                        "spearman_corr_avg": avg_spearman,
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
            "A",
        )
        progress_loss_sim, spearman_corr_sim = self._compute_progress_loss(
            progress_logits_ref_sim["B"],
            inputs["target_progress_A"],
            traj_sim_frames_shape,
            "sim",
        )

        progress_loss_diff, spearman_corr_diff = self._compute_progress_loss(
            progress_logits_ref_diff["B"],
            inputs["target_progress_B"],
            traj_diff_frames_shape,
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
