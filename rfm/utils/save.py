import os
import re
import shutil
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import gc
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from accelerate.state import AcceleratorState
from huggingface_hub import HfApi
from .upload_to_hub import upload_model_to_hub
from rfm.utils.distributed import rank_0_print, is_rank_0


class SaveBestCallback(TrainerCallback):
    """
    Save a checkpoint whenever `metric_name` improves.
    Works in DDP/accelerate: only rank 0 writes checkpoints.
    Optionally keeps top-k best checkpoints and uploads to Hub.
    Also saves a 'latest' checkpoint at regular intervals.
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        greater_is_better: List[bool] = None,
        keep_top_k: int = 1,
        save_every: Optional[int] = None,
        upload_to_hub: bool = False,
        hub_token: Optional[str] = None,
        hub_private: bool = False,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()
        self.metric_names = metric_names or ["custom_eval/p_rank_spearman_mw"]
        self.greater_is_better = greater_is_better or [True]

        # Validate inputs
        if len(self.metric_names) != len(self.greater_is_better):
            raise ValueError(
                f"metric_names ({len(self.metric_names)}) and greater_is_better ({len(self.greater_is_better)}) must have the same length"
            )
        self.keep_top_k = keep_top_k
        self.save_every = save_every
        self.upload_to_hub = upload_to_hub
        self.hub_token = hub_token
        self.hub_private = hub_private
        self.base_model = base_model
        self._best_val = None
        self._saved: List[Tuple[float, str]] = []  # list of (score, path), sorted from best -> worst
        self._uploaded: List[
            Tuple[float, str, str]
        ] = []  # list of (score, tag_name, commit_id), sorted from best -> worst
        self._trainer = None  # Will be set when callback is registered
        self._last_save_step = -1  # Track last step where we saved 'latest'
        self._previous_latest_ckpt_dir = None  # Track previous 'latest' checkpoint directory
        self._previous_latest_hub_tag = None  # Track previous 'latest' Hub tag

    def setup_trainer_reference(self, trainer: Trainer):
        """Set the trainer reference for later use in callbacks"""
        self._trainer = trainer

    def _compute_averaged_score(self, metrics: dict) -> Tuple[float, List[str]]:
        """
        Compute averaged score from multiple metrics.

        Returns:
            tuple: (averaged_score, missing_metrics)
        """
        available_scores = []
        missing_metrics = []

        for metric_name, is_better in zip(self.metric_names, self.greater_is_better):
            if metric_name in metrics:
                score = float(metrics[metric_name])
                # Normalize score: if lower is better, negate it so higher normalized score is better
                normalized_score = score if is_better else -score
                available_scores.append(normalized_score)
            else:
                missing_metrics.append(metric_name)

        if not available_scores:
            return float("-inf"), missing_metrics

        # Return average of normalized scores
        return np.mean(available_scores), missing_metrics

    def _is_main_process(self, trainer: Trainer) -> bool:
        try:
            return trainer.is_world_process_zero() and is_rank_0()
        except:
            return (not AcceleratorState().distributed_type) or AcceleratorState().is_main_process
    
    def _build_metric_short_name(self) -> str:
        """Build a short metric name for checkpoint naming."""
        if len(self.metric_names) == 1:
            return self.metric_names[0].split("/")[-1]
        else:
            return f"avg-{len(self.metric_names)}metrics"
    
    def _build_metrics_detail_string(self, metrics: dict) -> str:
        """Build a detailed metrics string for logging."""
        metrics_detail = []
        for name in self.metric_names:
            if name in metrics:
                metrics_detail.append(f"{name}:{metrics[name]:.4f}")
        return " | ".join(metrics_detail) if metrics_detail else "no metrics"
    
    def _build_individual_scores_string(self, metrics: dict) -> str:
        """Build individual scores string for commit messages."""
        individual_scores = []
        for name in self.metric_names:
            if name in metrics:
                individual_scores.append(f"{name.split('/')[-1]}={metrics[name]:.4f}")
        return ", ".join(individual_scores) if individual_scores else "no metrics"
    
    def _get_hub_model_id(self, args: TrainingArguments) -> str:
        """Get the Hub model ID from output directory."""
        base_name = args.output_dir.split("/")[-1].replace("_", "-")
        base_name = re.sub(r"-+", "-", base_name)
        base_name = base_name.strip("-")
        return f"rewardfm/{base_name}"
    
    def _clean_tag_name(self, tag_name: str) -> str:
        """Clean tag name for HuggingFace repo naming requirements."""
        tag_name = tag_name.replace("_", "-").replace(",", "")
        tag_name = re.sub(r"-+", "-", tag_name)
        tag_name = tag_name.strip("-")
        return tag_name
    
    def _save_checkpoint_files(self, args: TrainingArguments, ckpt_dir: str, metrics: dict = None, step: int = None):
        """Save model, trainer state files, and metrics."""
        # ALL processes must call this for distributed training to work correctly
        self._trainer.save_model(ckpt_dir)
        if args.should_save:
            self._trainer.save_state()  # trainer_state.json etc. in output_dir
            # save the trainer_state.json to the actual checkpoint directory
            shutil.copy(os.path.join(args.output_dir, "trainer_state.json"), ckpt_dir)
        
        # Save metrics to JSON file (only on main process)
        if metrics is not None and self._is_main_process(self._trainer):
            metrics_file = os.path.join(ckpt_dir, "metrics.json")
            metrics_to_save = {
                "step": step,
                "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                           for k, v in metrics.items()}
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            rank_0_print(f"📊 Saved metrics to {metrics_file}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _upload_checkpoint_to_hub(
        self, 
        ckpt_dir: str, 
        hub_model_id: str, 
        tag_name: str, 
        commit_message: str
    ) -> Tuple[str, str]:
        """Upload checkpoint to Hub and return URL and commit ID."""
        hub_url, commit_id = upload_model_to_hub(
            model_dir=ckpt_dir,
            hub_model_id=hub_model_id,
            private=self.hub_private,
            token=self.hub_token,
            commit_message=commit_message,
            base_model=self.base_model,
            tag_name=tag_name,
        )
        return hub_url, commit_id

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        # Gather metrics across all processes if using distributed training
        if hasattr(self._trainer, "accelerator") and self._trainer.accelerator.num_processes > 1:
            # Convert metrics to tensors for gathering
            gathered_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Convert to tensor and gather
                    tensor_value = self._trainer.accelerator.gather_for_metrics(
                        torch.tensor(value, dtype=torch.float32, device=self._trainer.accelerator.device)
                    )
                    # Take the mean across all processes
                    gathered_metrics[key] = tensor_value.mean().item()
                else:
                    # For non-numeric values, just use the local value
                    gathered_metrics[key] = value
            metrics = gathered_metrics

        score, missing_metrics = self._compute_averaged_score(metrics)

        if missing_metrics:
            rank_0_print(f"⚠️ Metrics {missing_metrics} not found in evaluation metrics")
            rank_0_print(f"Available metrics: {metrics.keys()}")
            if score == float("-inf"):  # All metrics missing
                return

        improved = (self._best_val is None) or (score > self._best_val)

        # Check if this score is worth saving (top-k logic)
        should_save = False
        if len(self._saved) < self.keep_top_k:
            # We haven't reached top-k yet, always save
            should_save = True
        else:
            # Check if this score beats the worst in our top-k
            worst_score = self._saved[-1][0]  # Last item is worst (sorted best -> worst)
            should_save = score > worst_score  # Always use > since we normalized scores

        if should_save and self._trainer:
            # Update overall best for reference
            if improved:
                self._best_val = score

            # Make a descriptive dir name
            step = state.global_step
            metric_short = self._build_metric_short_name()
            tag = f"{metric_short}={score:.4f}_step={step}"
            ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")

            metrics_str = self._build_metrics_detail_string(metrics)
            rank_0_print(
                f"💾 Saving ckpt: {ckpt_dir} | avg_score: {score:.6f} | {metrics_str} (rank {len(self._saved) + 1}/{self.keep_top_k})"
            )

            # Save model, trainer state, and metrics
            self._save_checkpoint_files(args, ckpt_dir, metrics, step)
            self._cleanup_memory()

            # Only manage saved list and file operations on rank 0
            if self._is_main_process(self._trainer):
                # Add to saved list and sort (always best -> worst since we normalized scores)
                self._saved.append((score, ckpt_dir))
                self._saved.sort(key=lambda x: x[0], reverse=True)

                # Remove old checkpoint if we exceed keep_top_k
                if len(self._saved) > self.keep_top_k:
                    _, path_to_rm = self._saved.pop(-1)
                    rank_0_print(f"🗑️ Removing old checkpoint: {path_to_rm}")
                    if os.path.isdir(path_to_rm):
                        shutil.rmtree(path_to_rm, ignore_errors=True)

                if self.upload_to_hub:
                    hub_model_id = self._get_hub_model_id(args)
                    tag_name = self._clean_tag_name(f"best-{metric_short}-{score:.4f}-step-{step}")
                    individual_scores_str = self._build_individual_scores_string(metrics)
                    commit_message = f"Checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"

                    rank_0_print(f"🚀 Uploading to Hub: {hub_model_id}")

                    hub_url, commit_id = self._upload_checkpoint_to_hub(
                        ckpt_dir=ckpt_dir,
                        hub_model_id=hub_model_id,
                        tag_name=tag_name,
                        commit_message=commit_message,
                    )
                    rank_0_print(f"✅ Successfully uploaded to: {hub_url}")
                    rank_0_print(f"🏷️ Tagged as: {tag_name}")

                    # Add to uploaded list and sort (always best -> worst since we normalized scores)
                    self._uploaded.append((score, tag_name, commit_id))
                    self._uploaded.sort(key=lambda x: x[0], reverse=True)

                    # Remove old tags if we exceed keep_top_k
                    api = HfApi(token=self.hub_token)
                    if len(self._uploaded) > self.keep_top_k:
                        _, old_tag, _ = self._uploaded.pop(-1)
                        rank_0_print(f"🗑️ Removing old Hub tag: {old_tag}")
                        api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=old_tag)
                        rank_0_print(f"✅ Deleted tag: {old_tag}")

                    # Aggressive memory cleanup after upload to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    rank_0_print("🧹 Cleaned up memory after Hub upload")

        # Save 'latest' checkpoint if save_every is configured and it's time to save
        # Do this after processing best checkpoints so we have the gathered metrics
        if self.save_every is not None and state.global_step > 0 and state.global_step % self.save_every == 0:
            if state.global_step != self._last_save_step:
                self._save_latest_checkpoint(args, state, metrics)
                self._last_save_step = state.global_step

        # Additional cleanup on all ranks after the entire on_evaluate callback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return control
    
    def _save_latest_checkpoint(self, args: TrainingArguments, state: TrainerState, metrics: dict):
        """Save a 'latest' checkpoint with metrics and step in the tag.
        Tracks and deletes the previous 'latest' checkpoint.
        
        Args:
            args: Training arguments
            state: Trainer state
            metrics: Evaluation metrics dictionary
        """
        if not self._trainer:
            return
        
        # Compute score and build tag similar to best checkpoints
        score, missing_metrics = self._compute_averaged_score(metrics)
        step = state.global_step
        metric_short = self._build_metric_short_name()
        
        # Build tag with metrics and step
        tag = f"latest-{metric_short}={score:.4f}_step={step}"
        ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")
        
        metrics_str = self._build_metrics_detail_string(metrics)
        rank_0_print(f"💾 Saving 'latest' checkpoint at step {step} to {ckpt_dir} | {metrics_str}")
        
        # Remove old 'latest' checkpoint if it exists (only on rank 0)
        if self._is_main_process(self._trainer):
            if self._previous_latest_ckpt_dir and os.path.isdir(self._previous_latest_ckpt_dir):
                rank_0_print(f"🗑️ Removing previous 'latest' checkpoint: {self._previous_latest_ckpt_dir}")
                shutil.rmtree(self._previous_latest_ckpt_dir, ignore_errors=True)
        
        # Save model, trainer state, and metrics
        self._save_checkpoint_files(args, ckpt_dir, metrics, step)
        rank_0_print(f"✅ Saved 'latest' checkpoint at step {step}")
        
        # Upload to Hub if configured
        if self.upload_to_hub and self._is_main_process(self._trainer):
            hub_model_id = self._get_hub_model_id(args)
            tag_name = self._clean_tag_name(f"latest-{metric_short}-{score:.4f}-step-{step}")
            individual_scores_str = self._build_individual_scores_string(metrics)
            commit_message = f"Latest checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"
            
            # Delete previous 'latest' Hub tag if it exists
            api = HfApi(token=self.hub_token)
            if self._previous_latest_hub_tag:
                try:
                    rank_0_print(f"🗑️ Removing previous 'latest' Hub tag: {self._previous_latest_hub_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=self._previous_latest_hub_tag)
                    rank_0_print(f"✅ Deleted previous tag: {self._previous_latest_hub_tag}")
                except Exception as e:
                    rank_0_print(f"⚠️ Could not delete previous Hub tag {self._previous_latest_hub_tag}: {e}")
            
            rank_0_print(f"🚀 Uploading 'latest' checkpoint to Hub: {hub_model_id}")
            
            hub_url, commit_id = self._upload_checkpoint_to_hub(
                ckpt_dir=ckpt_dir,
                hub_model_id=hub_model_id,
                tag_name=tag_name,
                commit_message=commit_message,
            )
            rank_0_print(f"✅ Successfully uploaded 'latest' to: {hub_url}")
            rank_0_print(f"🏷️ Tagged as: {tag_name}")
            
            # Track this as the new previous latest
            self._previous_latest_hub_tag = tag_name
        
        # Track this as the new previous latest checkpoint directory
        if self._is_main_process(self._trainer):
            self._previous_latest_ckpt_dir = ckpt_dir
        
        # Memory cleanup after saving model
        self._cleanup_memory()
