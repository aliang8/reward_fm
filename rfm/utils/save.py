import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
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
        metric_name: str = "custom_eval/p_rank_spearman_mw", 
        greater_is_better: bool = True, 
        keep_top_k: int = 1,
        upload_to_hub: bool = False,
        hub_token: Optional[str] = None,
        hub_private: bool = False,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    ):
        super().__init__()
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
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

    def _is_main_process(self, trainer: Trainer) -> bool:
        try:
            return trainer.is_world_process_zero()
        except:
            return (not AcceleratorState().distributed_type) or AcceleratorState().is_main_process

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):        
        if self.metric_name not in metrics:
            print(f"⚠️ Metric '{self.metric_name}' not found in evaluation metrics")
            return

        score = float(metrics[self.metric_name])
        improved = (
            (self._best_val is None) or
            (score > self._best_val if self.greater_is_better else score < self._best_val)
        )

        if improved and self._trainer and self._is_main_process(self._trainer):
            self._best_val = score

            # Make a descriptive dir name
            step = state.global_step
            metric_short = self.metric_name.split("/")[-1]  # Get the last part of metric name
            tag = f"{metric_short}={score:.4f}_step={step}"
            ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")
            print(f"Saving ckpt: {ckpt_dir}, new best {self.metric_name}: {score:.6f}")

            # Save model, optimizer/scheduler, RNG state, etc.
            self._trainer.save_model(ckpt_dir)
            if args.should_save:
                self._trainer.save_state()  # trainer_state.json etc. in output_dir

            # Upload to HF hub
            if self.upload_to_hub:
                base_name = args.output_dir.split("/")[-1].replace("_", "-")
                # Clean the tag name for HuggingFace repo naming requirements
                tag_name = f"best-{metric_short}-{score:.4f}-step-{step}".replace("_", "-").replace(",", "")
                hub_model_id = f"aliangdw/{base_name}"

                print(f"🚀 Uploading best model to Hub: {hub_model_id}")
            
                hub_url, commit_id = upload_model_to_hub(
                    model_dir=ckpt_dir,
                    hub_model_id=hub_model_id,
                    private=self.hub_private,
                    token=self.hub_token,
                    commit_message=f"Best checkpoint: {self.metric_name}={score:.4f} at step {step}",
                    base_model=self.base_model,
                    tag_name=tag_name
                )
                print(f"✅ Successfully uploaded to: {hub_url}")
                print(f"🏷️ Tagged as: {tag_name}")
                
                # Track uploaded models with tags and commit IDs from best -> worst
                self._uploaded.append((score, tag_name, commit_id))
                # Sort from best -> worst  
                reverse = self.greater_is_better
                self._uploaded.sort(key=lambda x: x[0], reverse=reverse)
                
                # Remove old tags beyond keep_top_k
                api = HfApi(token=self.hub_token)
                while len(self._uploaded) > self.keep_top_k:
                    _, old_tag, _ = self._uploaded.pop(-1)
                    try:
                        print(f"🗑️ Removing old Hub tag: {old_tag}")
                        api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=old_tag)
                        print(f"✅ Deleted tag: {old_tag}")
                    except Exception as e:
                        print(f"⚠️ Could not remove old Hub tag {old_tag}: {e}")

            # Track and prune
            self._saved.append((score, ckpt_dir))
            # Sort from best -> worst
            reverse = self.greater_is_better
            self._saved.sort(key=lambda x: x[0], reverse=reverse)

            while len(self._saved) > self.keep_top_k:
                _, path_to_rm = self._saved.pop(-1)
                print(f"🗑️ Removing old checkpoint: {path_to_rm}")
                if os.path.isdir(path_to_rm):
                    shutil.rmtree(path_to_rm, ignore_errors=True)

        return control