import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import List, Dict, Optional, Union, Any
import numpy as np

from rfm.utils.logging import is_rank_0, rank_0_print


def compute_metrics(eval_prediction):
    """
    Compute metrics for RFM evaluation across all three prediction types.
    This function is passed to the Trainer.
    """
    print(f"DEBUG: compute_metrics called with eval_prediction: {type(eval_prediction)}")
    print(f"DEBUG: eval_prediction.predictions shape: {eval_prediction.predictions.shape if eval_prediction.predictions is not None else None}")
    print(f"DEBUG: eval_prediction.label_ids shape: {eval_prediction.label_ids.shape if eval_prediction.label_ids is not None else None}")
    
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    
    if predictions is not None and len(predictions.shape) >= 2:
        # predictions should be [batch_size, 2] for all prediction types
        score_1 = predictions[:, 0]
        score_2 = predictions[:, 1]
        
        # Determine prediction type based on the context (this would need to be passed in metadata)
        # For now, we'll compute metrics for all types and let the user interpret them
        
        # Progress prediction metrics (score_1 = predicted, score_2 = target)
        progress_mse = ((score_1 - score_2) ** 2).mean()
        progress_mae = abs(score_1 - score_2).mean()
        
        # Preference prediction metrics (score_1 = A, score_2 = B)
        if label_ids is not None:
            # If we have preference labels, compute preference accuracy
            preference_logits = score_1 - score_2
            preference_probs = 1 / (1 + np.exp(-preference_logits))
            predicted_preferences = (preference_probs > 0.5).astype(float)
            preference_accuracy = (predicted_preferences == label_ids.astype(float)).mean()
        else:
            preference_accuracy = None
        
        # Similarity prediction metrics (score_1 = ref_A, score_2 = ref_B)
        # For similarity, we compare how similar trajectory A vs B are to the reference
        similarity_accuracy = (score_1 > score_2).astype(float).mean()  # A more similar than B
        similarity_diff = score_1 - score_2
        
        metrics = {
            # Progress metrics
            "progress_mse": progress_mse,
            "progress_mae": progress_mae,
            
            # Preference metrics
            "preference_accuracy": preference_accuracy,
            "avg_score_A": score_1.mean(),
            "avg_score_B": score_2.mean(),
            
            # Similarity metrics
            "similarity_accuracy": similarity_accuracy,
            "similarity_diff": similarity_diff.mean(),
            "avg_score_ref_A": score_1.mean(),
            "avg_score_ref_B": score_2.mean(),
        }
        
        print(f"DEBUG: computed metrics: {metrics}")
        return metrics
    else:
        print(f"DEBUG: predictions is None or wrong shape: {predictions}")
        return {}
        
class RFMTrainer(Trainer):
    def __init__(self, *args, beta=0.1, compute_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.compute_metrics = compute_metrics
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

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
        }

        # Add individual loss components
        if "preference_loss" in aggregated_losses:
            log_data["train/preference_loss"] = aggregated_losses["preference_loss"]
        
        if "similarity_loss" in aggregated_losses:
            log_data["train/similarity_loss"] = aggregated_losses["similarity_loss"]
        
        if "progress_loss" in aggregated_losses:
            log_data["train/progress_loss"] = aggregated_losses["progress_loss"]

        # Add batch composition info
        if "preference_count" in aggregated_losses:
            log_data["train/preference_samples"] = aggregated_losses["preference_count"]
        
        if "similarity_count" in aggregated_losses:
            log_data["train/similarity_samples"] = aggregated_losses["similarity_count"]
        
        if "progress_count" in aggregated_losses:
            log_data["train/progress_samples"] = aggregated_losses["progress_count"]

        # Calculate batch composition percentages
        total_samples = aggregated_losses.get("preference_count", 0) + aggregated_losses.get("similarity_count", 0)
        if total_samples > 0:
            log_data["train/pref_ratio"] = aggregated_losses.get("preference_count", 0) / total_samples
            log_data["train/sim_ratio"] = aggregated_losses.get("similarity_count", 0) / total_samples

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
            if "preference_loss" in aggregated_losses:
                rank_0_print(f"  Preference Loss: {aggregated_losses['preference_loss']:.6f}")
            if "similarity_loss" in aggregated_losses:
                rank_0_print(f"  Similarity Loss: {aggregated_losses['similarity_loss']:.6f}")
            if "progress_loss" in aggregated_losses:
                rank_0_print(f"  Progress Loss: {aggregated_losses['progress_loss']:.6f}")
            if total_samples > 0:
                rank_0_print(f"  Batch Composition: {aggregated_losses.get('preference_count', 0)} pref, {aggregated_losses.get('similarity_count', 0)} sim")

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
        loss_keys = ["preference_loss", "similarity_loss", "progress_loss"]
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

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """
        Override prediction_step to handle RFM-specific input structure during evaluation.
        """
        model.eval()
        
        # Extract the separate batches (same as in compute_loss)
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        
        # If we have no properly structured inputs, fall back to default behavior
        if not preference_inputs and not similarity_inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        losses = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            # Process preference inputs if available
            if num_preferences > 0 and preference_inputs:
                pref_loss, pref_outputs = self._compute_preference_loss(model, preference_inputs, return_outputs=True)
                losses.append(pref_loss.detach())
                
                # Extract logits and labels for metrics computation
                if "preference_scores" in pref_outputs:
                    # Preference scores are [batch_size], reshape to [batch_size, 1] for consistency
                    pref_logits = pref_outputs["preference_scores"].detach()
                    if pref_logits.dim() == 1:
                        pref_logits = pref_logits.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
                    all_logits.append(pref_logits)
                if "preference_labels" in pref_outputs:
                    all_labels.append(pref_outputs["preference_labels"].detach())
            
            # Process similarity inputs if available
            if num_similarities > 0 and similarity_inputs:
                sim_loss, sim_outputs = self._compute_similarity_loss(model, similarity_inputs, return_outputs=True)
                losses.append(sim_loss.detach())
                
                # Extract logits for metrics computation
                if "score_ref_A" in sim_outputs and "score_ref_B" in sim_outputs:
                    # Stack A and B scores for metrics computation -> [batch_size, 2]
                    score_A = sim_outputs["score_ref_A"].detach()
                    score_B = sim_outputs["score_ref_B"].detach()
                    
                    # Ensure both scores are [batch_size, 1] before stacking
                    if score_A.dim() == 1:
                        score_A = score_A.unsqueeze(-1)
                    if score_B.dim() == 1:
                        score_B = score_B.unsqueeze(-1)
                    
                    sim_logits = torch.cat([score_A, score_B], dim=1)  # [batch_size, 2]
                    all_logits.append(sim_logits)
        
        # Combine losses
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=model.device)
        
        # Combine logits and labels
        if all_logits:
            combined_logits = torch.cat(all_logits, dim=0)
        else:
            combined_logits = None
            
        if all_labels:
            combined_labels = torch.cat(all_labels, dim=0)
        else:
            combined_labels = None
        
        return (total_loss, combined_logits, combined_labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for separate preference and similarity batches."""
        
        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        
        # Initialize loss components and metadata
        total_loss = 0.0
        loss_metadata = {
            "preference_loss": 0.0, 
            "similarity_loss": 0.0,
            "progress_loss": 0.0,
            "preference_count": num_preferences,
            "similarity_count": num_similarities,
            "progress_count": 0
        }
        
        # Compute preference loss if we have preference samples
        if num_preferences > 0 and preference_inputs:
            preference_loss, preference_outputs = self._compute_preference_loss(model, preference_inputs, return_outputs=True)
            total_loss += preference_loss * num_preferences
            loss_metadata["preference_loss"] = preference_outputs.get("preference_loss", 0.0)
            loss_metadata["progress_loss"] += preference_outputs.get("progress_loss", 0.0) * num_preferences
            loss_metadata["progress_count"] += num_preferences
        
        # Compute similarity loss if we have similarity samples
        if num_similarities > 0 and similarity_inputs:
            similarity_loss, similarity_outputs = self._compute_similarity_loss(model, similarity_inputs, return_outputs=True)
            total_loss += similarity_loss * num_similarities
            loss_metadata["similarity_loss"] = similarity_outputs.get("similarity_loss", 0.0)
            loss_metadata["progress_loss"] += (similarity_outputs.get("progress_loss_A", 0.0) + similarity_outputs.get("progress_loss_B", 0.0)) * num_similarities
            loss_metadata["progress_count"] += num_similarities * 2  # Both A and B for similarity
        
        # Normalize by total batch size
        total_batch_size = num_preferences + num_similarities
        if total_batch_size > 0:
            total_loss = total_loss / total_batch_size
        
        # Always store custom losses for logging (even when return_outputs=False)
        self.custom_losses = loss_metadata
        
        if return_outputs:
            # Combine outputs from all loss functions
            combined_outputs = {
                "loss_metadata": loss_metadata,
                "total_loss": total_loss,
                "batch_size": total_batch_size
            }
            return total_loss, combined_outputs
        
        return total_loss

    def _compute_progress_loss_from_outputs(self, progress_logits, target_progress_A, target_progress_B=None):
        """Helper function to compute progress loss from model outputs."""
        progress_loss = 0.0
        progress_metadata = {}

        # Compute progress loss for trajectory A if target_progress_A is provided
        if target_progress_A is not None and progress_logits is not None:
            predicted_progress_A = progress_logits.squeeze(-1)  # [batch_size]
            final_target_progress_A = target_progress_A[:, -1]  # [batch_size] - use final frame progress
            progress_loss_A = F.mse_loss(predicted_progress_A, final_target_progress_A)
            progress_loss += progress_loss_A
            progress_metadata["predicted_progress_A"] = predicted_progress_A
            progress_metadata["target_progress_A"] = final_target_progress_A
            progress_metadata["progress_loss_A"] = progress_loss_A.item()
        
        # Compute progress loss for trajectory B if target_progress_B is provided
        if target_progress_B is not None and progress_logits is not None:
            predicted_progress_B = progress_logits.squeeze(-1)  # [batch_size]
            final_target_progress_B = target_progress_B[:, -1]  # [batch_size] - use final frame progress
            progress_loss_B = F.mse_loss(predicted_progress_B, final_target_progress_B)
            progress_loss += progress_loss_B
            progress_metadata["predicted_progress_B"] = predicted_progress_B
            progress_metadata["target_progress_B"] = final_target_progress_B
            progress_metadata["progress_loss_B"] = progress_loss_B.item()
        
        return progress_loss, progress_metadata
    
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
            target_progress=inputs.get("target_progress_A"),  # Pass target progress for trajectory A
            prediction_type="preference"
        )
                
        # Get preference scores from the preference head
        preference_scores = model_outputs.logits.squeeze(-1)  # [batch_size]
        
        # Get preference labels (1 if A is preferred, 0 if B is preferred)
        preference_labels = inputs["preference_labels"]
        
        # Binary cross entropy loss for preference prediction
        preference_loss = F.binary_cross_entropy_with_logits(preference_scores, preference_labels.float())
        
        # Compute progress loss if progress_logits existe
        progress_loss, progress_metadata = self._compute_progress_loss_from_outputs(
            progress_logits, 
            inputs["target_progress_A"], 
            inputs["target_progress_B"]
        )
        
        # Combine losses
        total_loss = preference_loss + progress_loss
        
        if return_outputs:
            outputs_dict = {
                "preference_scores": preference_scores, 
                "preference_labels": preference_labels,
                "preference_loss": preference_loss.item(),
                "progress_loss": progress_loss.item() if isinstance(progress_loss, torch.Tensor) else progress_loss
            }
            outputs_dict.update(progress_metadata)
            return total_loss, outputs_dict
        return total_loss
    
    def _compute_similarity_loss(self, model, inputs, return_outputs=False):
        """Compute similarity scoring loss (DPO-style)."""
        # Forward pass for reference vs trajectory A
        model_outputs_ref_A, progress_logits_ref_A = model(
            input_ids=inputs["input_ids_ref_A"],
            attention_mask=inputs["attention_mask_ref_A"],
            pixel_values=inputs.get("pixel_values_ref_A"),
            pixel_values_videos=inputs.get("pixel_values_videos_ref_A"),
            image_grid_thw=inputs.get("image_grid_thw_ref_A"),
            video_grid_thw=inputs.get("video_grid_thw_ref_A"),
            target_progress=inputs.get("target_progress_A"),  # Pass target progress for trajectory A
            prediction_type="similarity"
        )
        
        # Forward pass for reference vs trajectory B
        model_outputs_ref_B, progress_logits_ref_B = model(
            input_ids=inputs["input_ids_ref_B"],
            attention_mask=inputs["attention_mask_ref_B"],
            pixel_values=inputs.get("pixel_values_ref_B"),
            pixel_values_videos=inputs.get("pixel_values_videos_ref_B"),
            image_grid_thw=inputs.get("image_grid_thw_ref_B"),
            video_grid_thw=inputs.get("video_grid_thw_ref_B"),
            target_progress=inputs.get("target_progress_B"),  # Pass target progress for trajectory B
            prediction_type="similarity"
        )
        
        # Extract similarity scores
        score_ref_A = model_outputs_ref_A.logits.squeeze(-1)
        score_ref_B = model_outputs_ref_B.logits.squeeze(-1)
        
        # Compute DPO-style loss: encourage trajectory A to be more similar to reference than trajectory B
        # This assumes trajectory A is the "better" trajectory (more similar to reference)
        similarity_loss = -F.logsigmoid(self.beta * (score_ref_A - score_ref_B)).mean()
        
        # Compute progress loss for both forward passes
        progress_loss_A, progress_metadata_A = self._compute_progress_loss_from_outputs(
            progress_logits_ref_A, 
            inputs["target_progress_A"]
        )
        progress_loss_B, progress_metadata_B = self._compute_progress_loss_from_outputs(
            progress_logits_ref_B, 
            inputs["target_progress_B"]
        )
        
        # Combine losses
        total_loss = similarity_loss + progress_loss_A + progress_loss_B
        
        if return_outputs:
            outputs_dict = {
                "score_ref_A": score_ref_A, 
                "score_ref_B": score_ref_B,
                "similarity_loss": similarity_loss.item(),
                "progress_loss_A": progress_loss_A.item() if isinstance(progress_loss_A, torch.Tensor) else progress_loss_A,
                "progress_loss_B": progress_loss_B.item() if isinstance(progress_loss_B, torch.Tensor) else progress_loss_B
            }
            # Combine progress metadata with A and B suffixes
            for key, value in progress_metadata_A.items():
                outputs_dict[f"{key}_A"] = value
            for key, value in progress_metadata_B.items():
                outputs_dict[f"{key}_B"] = value
            return total_loss, outputs_dict
        return total_loss