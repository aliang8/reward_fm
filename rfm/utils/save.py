import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from accelerate.state import AcceleratorState
from huggingface_hub import HfApi
from .upload_to_hub import upload_model_to_hub

class SaveBestCallback(TrainerCallback):
    """
    Save a checkpoint whenever `metric_name` improves.
    Works in DDP/accelerate: only rank 0 writes checkpoints.
    Optionally keeps top-k best checkpoints and uploads to Hub.
    """
    def __init__(
        self, 
        metric_names: List[str] = None, 
        greater_is_better: List[bool] = None, 
        keep_top_k: int = 1,
        upload_to_hub: bool = False,
        hub_token: Optional[str] = None,
        hub_private: bool = False,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    ):
        super().__init__()
        self.metric_names = metric_names or ["custom_eval/p_rank_spearman_mw"]
        self.greater_is_better = greater_is_better or [True]
        
        # Validate inputs
        if len(self.metric_names) != len(self.greater_is_better):
            raise ValueError(f"metric_names ({len(self.metric_names)}) and greater_is_better ({len(self.greater_is_better)}) must have the same length")
        self.keep_top_k = keep_top_k
        self.upload_to_hub = upload_to_hub
        self.hub_token = hub_token
        self.hub_private = hub_private
        self.base_model = base_model
        self._best_val = None
        self._saved: List[Tuple[float, str]] = []  # list of (score, path), sorted from best -> worst
        self._uploaded: List[Tuple[float, str, str]] = []  # list of (score, tag_name, commit_id), sorted from best -> worst
        self._trainer = None  # Will be set when callback is registered

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
            return float('-inf'), missing_metrics
            
        # Return average of normalized scores
        return np.mean(available_scores), missing_metrics

    def _is_main_process(self, trainer: Trainer) -> bool:
        try:
            return trainer.is_world_process_zero()
        except:
            return (not AcceleratorState().distributed_type) or AcceleratorState().is_main_process

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):        
        score, missing_metrics = self._compute_averaged_score(metrics)
        
        if missing_metrics:
            print(f"⚠️ Metrics {missing_metrics} not found in evaluation metrics")
            if score == float('-inf'):  # All metrics missing
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

        if should_save and self._trainer and self._is_main_process(self._trainer):
            # Update overall best for reference
            if improved:
                self._best_val = score

            # Make a descriptive dir name
            step = state.global_step
            # Use "avg" if multiple metrics, otherwise use the single metric name
            if len(self.metric_names) == 1:
                metric_short = self.metric_names[0].split("/")[-1]
            else:
                metric_short = f"avg-{len(self.metric_names)}metrics"
            
            tag = f"{metric_short}={score:.4f}_step={step}"
            ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")
            
            # Build detailed metrics string for logging
            metrics_detail = []
            for name in self.metric_names:
                if name in metrics:
                    metrics_detail.append(f"{name}:{metrics[name]:.4f}")
            metrics_str = " | ".join(metrics_detail)
            
            print(f"💾 Saving ckpt: {ckpt_dir} | avg_score: {score:.6f} | {metrics_str} (rank {len(self._saved)+1}/{self.keep_top_k})")

            # Save model, optimizer/scheduler, RNG state, etc.
            self._trainer.save_model(ckpt_dir)
            if args.should_save:
                self._trainer.save_state()  # trainer_state.json etc. in output_dir

            # Add to saved list and sort (always best -> worst since we normalized scores)
            self._saved.append((score, ckpt_dir))
            self._saved.sort(key=lambda x: x[0], reverse=True)

            # Remove old checkpoint if we exceed keep_top_k
            if len(self._saved) > self.keep_top_k:
                _, path_to_rm = self._saved.pop(-1)
                print(f"🗑️ Removing old checkpoint: {path_to_rm}")
                if os.path.isdir(path_to_rm):
                    shutil.rmtree(path_to_rm, ignore_errors=True)

            # Upload to HF hub
            if self.upload_to_hub:
                base_name = args.output_dir.split("/")[-1].replace("_", "-")
                # Clean the tag name for HuggingFace repo naming requirements
                tag_name = f"best-{metric_short}-{score:.4f}-step-{step}".replace("_", "-").replace(",", "")
                hub_model_id = f"aliangdw/{base_name}"

                # Build detailed commit message with individual metric scores
                individual_scores = []
                for name in self.metric_names:
                    if name in metrics:
                        individual_scores.append(f"{name.split('/')[-1]}={metrics[name]:.4f}")
                individual_scores_str = ", ".join(individual_scores)
                
                commit_message = f"Checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"

                print(f"🚀 Uploading to Hub: {hub_model_id}")
            
                hub_url, commit_id = upload_model_to_hub(
                    model_dir=ckpt_dir,
                    hub_model_id=hub_model_id,
                    private=self.hub_private,
                    token=self.hub_token,
                    commit_message=commit_message,
                    base_model=self.base_model,
                    tag_name=tag_name
                )
                print(f"✅ Successfully uploaded to: {hub_url}")
                print(f"🏷️ Tagged as: {tag_name}")
                
                # Add to uploaded list and sort (always best -> worst since we normalized scores)
                self._uploaded.append((score, tag_name, commit_id))
                self._uploaded.sort(key=lambda x: x[0], reverse=True)
                
                # Remove old tags if we exceed keep_top_k
                api = HfApi(token=self.hub_token)
                if len(self._uploaded) > self.keep_top_k:
                    _, old_tag, _ = self._uploaded.pop(-1)
                    print(f"🗑️ Removing old Hub tag: {old_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=old_tag)
                    print(f"✅ Deleted tag: {old_tag}")

        return control