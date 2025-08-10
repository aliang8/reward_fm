import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import List, Dict, Optional, Union, Any
import numpy as np

from rfm.utils.logging import is_rank_0, rank_0_print
from rfm.utils.metrics import compute_spearman_correlation, compute_auc, compute_mse, compute_mae


def compute_metrics(eval_prediction):
    """
    Compute metrics for RFM evaluation across all three prediction types.
    This function is passed to the Trainer.
    
    Note: This function is now a fallback. The main evaluation is handled
    by the custom evaluate() method in RFMTrainer.
    """
    print(f"DEBUG: compute_metrics called with eval_prediction: {type(eval_prediction)}")
    print(f"DEBUG: eval_prediction.predictions shape: {eval_prediction.predictions.shape if eval_prediction.predictions is not None else None}")
    print(f"DEBUG: eval_prediction.label_ids shape: {eval_prediction.label_ids.shape if eval_prediction.label_ids is not None else None}")
    
    # This is a fallback - the main evaluation is now handled by the custom evaluate() method
    # which provides more comprehensive RFM-specific metrics
    
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    
    if predictions is not None and len(predictions.shape) >= 2:
        # predictions should be [batch_size, 2] for all prediction types
        score_1 = predictions[:, 0]
        score_2 = predictions[:, 1]
        
        # Basic metrics as fallback
        progress_mse = ((score_1 - score_2) ** 2).mean()
        progress_mae = abs(score_1 - score_2).mean()
        
        metrics = {
            "fallback_progress_mse": progress_mse,
            "fallback_progress_mae": progress_mae,
            "note": "Using fallback metrics. Use trainer.evaluate() for full RFM metrics."
        }
        
        print(f"DEBUG: computed fallback metrics: {metrics}")
        return metrics
    else:
        print(f"DEBUG: predictions is None or wrong shape: {predictions}")
        return {"note": "No predictions available for fallback metrics"}
        
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
        
        return total_loss, None, None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
        # Get the evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Initialize metrics storage
        all_preference_scores = []
        all_preference_labels = []
        all_progress_predictions = []
        all_progress_targets = []
        all_similarity_scores_A = []
        all_similarity_scores_B = []
        
        # Set model to eval mode
        self.model.eval()
        
        import ipdb; ipdb.set_trace()
        # Run evaluation
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Move inputs to device
                inputs = self._prepare_inputs(inputs)
                
                # Extract batch information
                preference_inputs = inputs.get("preference_inputs", {})
                similarity_inputs = inputs.get("similarity_inputs", {})
                num_preferences = inputs.get("num_preferences", 0)
                num_similarities = inputs.get("num_similarities", 0)
                
                # Process preference inputs
                if num_preferences > 0 and preference_inputs:
                    _, pref_outputs = self._compute_preference_loss(self.model, preference_inputs, return_outputs=True)
                    
                    # Store preference predictions and labels
                    if "preference_scores" in pref_outputs:
                        all_preference_scores.append(pref_outputs["preference_scores"].cpu())
                        all_preference_labels.append(pref_outputs["preference_labels"].cpu())
                    
                    # Store progress predictions and targets
                    if "predicted_progress_A" in pref_outputs:
                        # predicted_progress_A is always a list of tensors
                        progress_preds = torch.cat([p.cpu() for p in pref_outputs["predicted_progress_A"] if len(p) > 0], dim=0)
                        progress_targets = torch.cat([t.cpu() for t in pref_outputs["target_progress_A"] if len(t) > 0], dim=0)
                        all_progress_predictions.append(progress_preds)
                        all_progress_targets.append(progress_targets)
                
                # Process similarity inputs
                if num_similarities > 0 and similarity_inputs:
                    _, sim_outputs = self._compute_similarity_loss(self.model, similarity_inputs, return_outputs=True)
                    
                    # Store similarity scores
                    if "score_ref_A" in sim_outputs:
                        all_similarity_scores_A.append(sim_outputs["score_ref_A"].cpu())
                    if "score_ref_B" in sim_outputs:
                        all_similarity_scores_B.append(sim_outputs["score_ref_B"].cpu())
                    
                    # Store progress predictions and targets for both A and B
                    if "predicted_progress_A" in sim_outputs:
                        # predicted_progress_A is always a list of tensors
                        progress_preds_A = torch.cat([p.cpu() for p in sim_outputs["predicted_progress_A"] if len(p) > 0], dim=0)
                        progress_targets_A = torch.cat([t.cpu() for t in sim_outputs["target_progress_A"] if len(t) > 0], dim=0)
                        all_progress_predictions.append(progress_preds_A)
                        all_progress_targets.append(progress_targets_A)
                    
                    if "predicted_progress_B" in sim_outputs:
                        # predicted_progress_B is always a list of tensors
                        progress_preds_B = torch.cat([p.cpu() for p in sim_outputs["predicted_progress_B"] if len(p) > 0], dim=0)
                        progress_targets_B = torch.cat([t.cpu() for t in sim_outputs["target_progress_B"] if len(t) > 0], dim=0)
                        all_progress_predictions.append(progress_preds_B)
                        all_progress_targets.append(progress_targets_B)
        
        # Compute metrics
        metrics = {}
        
        # 1. Preference Accuracy
        if all_preference_scores and all_preference_labels:
            preference_scores = torch.cat(all_preference_scores, dim=0)
            preference_labels = torch.cat(all_preference_labels, dim=0)
            
            # Convert logits to probabilities and predictions
            preference_probs = torch.sigmoid(preference_scores)
            preference_predictions = (preference_probs > 0.5).float()
            
            # Compute accuracy
            preference_accuracy = (preference_predictions == preference_labels).float().mean().item()
            metrics[f"{metric_key_prefix}_preference_accuracy"] = preference_accuracy
            
            # Additional preference metrics
            metrics[f"{metric_key_prefix}_preference_auc"] = compute_auc(preference_scores, preference_labels)
            metrics[f"{metric_key_prefix}_avg_preference_score"] = preference_scores.mean().item()
        
        # 2. Progress Prediction - Spearman Correlation
        if all_progress_predictions and all_progress_targets:
            progress_predictions = torch.cat(all_progress_predictions, dim=0)  # [total_samples, seq_len]
            progress_targets = torch.cat(all_progress_targets, dim=0)  # [total_samples, seq_len]
            
            # Compute Spearman correlation for each sample (temporal correlation)
            spearman_correlations = []
            for pred, target in zip(progress_predictions, progress_targets):
                if len(pred) > 1 and len(target) > 1:  # Need at least 2 points for correlation
                    corr = compute_spearman_correlation(pred, target)
                    if not torch.isnan(corr):
                        spearman_correlations.append(corr.item())
            
            if spearman_correlations:
                mean_spearman = np.mean(spearman_correlations)
                std_spearman = np.std(spearman_correlations)
                metrics[f"{metric_key_prefix}_progress_spearman_mean"] = mean_spearman
                metrics[f"{metric_key_prefix}_progress_spearman_std"] = std_spearman
                metrics[f"{metric_key_prefix}_progress_spearman_median"] = np.median(spearman_correlations)
                
                # Also compute MSE and MAE for progress across all timesteps
                progress_mse = compute_mse(progress_predictions, progress_targets)
                progress_mae = compute_mae(progress_predictions, progress_targets)
                metrics[f"{metric_key_prefix}_progress_mse"] = progress_mse
                metrics[f"{metric_key_prefix}_progress_mae"] = progress_mae
                
                # Additional progress metrics
                metrics[f"{metric_key_prefix}_progress_samples"] = len(spearman_correlations)
                metrics[f"{metric_key_prefix}_progress_seq_length"] = progress_predictions.shape[1]
        
        # 3. Similarity Metrics
        if all_similarity_scores_A and all_similarity_scores_B:
            similarity_scores_A = torch.cat(all_similarity_scores_A, dim=0)
            similarity_scores_B = torch.cat(all_similarity_scores_B, dim=0)
            
            # Compute similarity difference (A should be higher than B)
            similarity_diff = similarity_scores_A - similarity_scores_B
            metrics[f"{metric_key_prefix}_similarity_diff_mean"] = similarity_diff.mean().item()
            metrics[f"{metric_key_prefix}_similarity_diff_std"] = similarity_diff.std().item()
            
            # Compute accuracy: how often A is scored higher than B
            similarity_accuracy = (similarity_diff > 0).float().mean().item()
            metrics[f"{metric_key_prefix}_similarity_accuracy"] = similarity_accuracy
        
        # 4. Overall evaluation loss
        eval_loss = self.compute_loss(self.model, inputs)
        metrics[f"{metric_key_prefix}_loss"] = eval_loss.item()
        
        # Log metrics
        if is_rank_0():
            rank_0_print(f"\n=== Custom RFM Evaluation Results ===")
            for key, value in metrics.items():
                rank_0_print(f"{key}: {value:.6f}")
            rank_0_print("=" * 50)
        
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
        """
        Helper function to compute progress loss from model outputs.
        
        Args:
            progress_logits: Dict with 'A' and 'B' keys containing progress predictions as tensors
            target_progress_A: Target progress for trajectory A (tensor)
            target_progress_B: Target progress for trajectory B (tensor, optional)
            
        Returns:
            tuple: (total_progress_loss, progress_metadata)
        """
        progress_loss = 0.0
        progress_metadata = {}

        # progress_logits is always a dict with 'A' and 'B' keys
        progress_logits_A = progress_logits['A']  # [batch_size, seq_len]
        progress_logits_B = progress_logits['B']  # [batch_size, seq_len]
        
        # Compute progress loss for trajectory A
        if target_progress_A is not None:
            # Compute MSE loss directly on tensors
            progress_loss_A = F.mse_loss(progress_logits_A, target_progress_A)
            progress_loss += progress_loss_A
            
            progress_metadata["predicted_progress_A"] = progress_logits_A
            progress_metadata["target_progress_A"] = target_progress_A
            progress_metadata["progress_loss_A"] = progress_loss_A.item()
        
        # Compute progress loss for trajectory B
        if target_progress_B is not None:
            # Compute MSE loss directly on tensors
            progress_loss_B = F.mse_loss(progress_logits_B, target_progress_B)
            progress_loss += progress_loss_B
            
            progress_metadata["predicted_progress_B"] = progress_logits_B
            progress_metadata["target_progress_B"] = target_progress_B
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
        
        # Compute progress loss if progress_logits exist
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