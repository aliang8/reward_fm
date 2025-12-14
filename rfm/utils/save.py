import os
import re
import shutil
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime
import numpy as np
import gc
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from accelerate.state import AcceleratorState
from huggingface_hub import HfApi, snapshot_download
from .upload_to_hub import upload_model_to_hub
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import loguru_logger as logger


def resolve_checkpoint_path(checkpoint_path: Optional[str], hub_token: Optional[str] = None) -> Optional[str]:
    """
    Resolve checkpoint path, supporting local paths and HuggingFace Hub with @ notation.

    Args:
        checkpoint_path: Path to checkpoint. Can be:
            - None: No checkpoint to load
            - Local path: /path/to/checkpoint
            - HF repo: username/model-name (loads best tag automatically)
            - HF repo with tag: username/model-name@tag-name
        hub_token: Optional HuggingFace token for private repos

    Returns:
        Resolved local path to checkpoint, or None if no checkpoint
    """
    if not checkpoint_path:
        return None

    # If it's a local path, return as-is
    if checkpoint_path.startswith("/") or checkpoint_path.startswith("./") or checkpoint_path.startswith("../"):
        logger.info(f"Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    # Check if it looks like a HuggingFace repo (contains /)
    if "/" in checkpoint_path:
        repo_id, revision = parse_hf_model_id_and_revision(checkpoint_path, model_name="checkpoint")

        # Download from HuggingFace Hub
        logger.info(f"Downloading checkpoint from HuggingFace Hub: {repo_id}@{revision or 'latest'}")
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=hub_token,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "*.yaml"],
        )
        logger.info(f"Downloaded checkpoint to: {local_path}")
        return local_path

    # Otherwise, treat as local path
    logger.info(f"Using checkpoint: {checkpoint_path}")
    return checkpoint_path


def parse_hf_model_id_and_revision(hf_model_id: str, model_name: str = "model") -> Tuple[str, Optional[str]]:
    """
    Parse HuggingFace model ID and determine which revision (tag) to load.

    Supports explicit revisions via repo@revision format, or automatically
    finds the best tag if no explicit revision is provided.

    Args:
        hf_model_id: HuggingFace model repository ID or local path, optionally with @revision
        model_name: Name of the model type for logging (e.g., "ReWiND model", "Qwen model")

    Returns:
        Tuple of (repo_id, revision_to_load) where:
        - repo_id: The repository ID without the @revision suffix
        - revision_to_load: The revision/tag to load, or None for latest
    """
    # Allow users to specify explicit revisions via repo@revision
    if "@" in hf_model_id:
        repo_id, explicit_revision = hf_model_id.split("@", 1)
    else:
        repo_id, explicit_revision = hf_model_id, None

    revision_to_load = explicit_revision

    # Check if this is a HuggingFace repo (not a local path) and find best tag
    if "/" in repo_id and not repo_id.startswith("/"):
        if revision_to_load:
            logger.info(f"Loading {model_name} {repo_id} at explicit revision '{revision_to_load}'")
        else:
            best_tag, best_score = find_best_model_tag(repo_id)
            if best_tag:
                revision_to_load = best_tag
                logger.info(f"Loading {model_name} from best tag: {repo_id}@{revision_to_load} (score: {best_score})")
            else:
                logger.info(f"No best tag found, loading latest revision of {repo_id}")
    else:
        logger.info(f"Loading local/explicit {model_name} from {repo_id}")

    return repo_id, revision_to_load


def find_best_model_tag(hf_model_id: str, hub_token: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the best model tag from HuggingFace Hub by parsing tag names and extracting scores.

    Expected tag format: "best-{metric_short}-{score:.4f}-step-{step}"
    Example: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"

    Args:
        hf_model_id: HuggingFace model ID (e.g., "aliangdw/rewind-debug")
        hub_token: Optional HuggingFace token for private repos

    Returns:
        tuple: (best_tag_name, best_score) or (None, None) if no valid tags found
    """
    try:
        api = HfApi(token=hub_token)

        # Check if repository exists
        if not api.repo_exists(repo_id=hf_model_id, repo_type="model"):
            logger.info(f"Repository {hf_model_id} does not exist")
            return None, None

        # Get all tags for the repository
        tags = api.list_repo_refs(repo_id=hf_model_id, repo_type="model").tags

        if not tags:
            logger.info(f"No tags found in repository {hf_model_id}")
            return None, None

        logger.info(f"Found {len(tags)} tags in {hf_model_id}: {[tag.name for tag in tags]}")

        best_tag = None
        best_score = float("-inf")

        # Parse each tag to extract score
        for tag in tags:
            tag_name = tag.name

            # Match our tag pattern: "best-{metric_short}-{score}-step-{step}"
            # Examples: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"
            # Score can be positive or negative (e.g., 0.8500 or -1.2300)
            pattern = r"best-.*?-(-?\d+\.\d+)-step-\d+"
            match = re.search(pattern, tag_name)

            if match:
                try:
                    score = float(match.group(1))
                    logger.info(f"Parsed tag '{tag_name}': score = {score}")

                    if score > best_score:
                        best_score = score
                        best_tag = tag_name

                except ValueError:
                    logger.info(f"Could not parse score from tag '{tag_name}'")
                    continue
            else:
                logger.info(f"Tag '{tag_name}' does not match expected pattern")

        if best_tag:
            logger.info(f"Best tag found: '{best_tag}' with score {best_score}")
        else:
            logger.info("No valid tags found matching the expected pattern")

        return best_tag, best_score

    except Exception as e:
        logger.info(f"Error finding best tag for {hf_model_id}: {e}")
        return None, None


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
        hub_save_every: Optional[int] = None,
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
        self.hub_save_every = hub_save_every  # Frequency for Hub uploads (None = upload every checkpoint)
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
        self._last_best_save_step = -1  # Track last step where we saved 'best'
        self._last_hub_upload_step = -1  # Track last step where we uploaded 'best' to Hub
        self._last_latest_hub_upload_step = -1  # Track last step where we uploaded 'latest' to Hub
        self._previous_latest_ckpt_dir = None  # Track previous 'latest' checkpoint directory
        self._previous_latest_hub_tag = None  # Track previous 'latest' Hub tag
        self._run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Static timestamp for this run

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
        """Get the Hub model ID from output directory with timestamp."""
        base_name = args.output_dir.split("/")[-1].replace("_", "-")
        base_name = re.sub(r"-+", "-", base_name)
        base_name = base_name.strip("-")
        return f"rewardfm/{base_name}-{self._run_timestamp}"

    def _clean_tag_name(self, tag_name: str) -> str:
        """Clean tag name for HuggingFace repo naming requirements."""
        tag_name = tag_name.replace("_", "-").replace(",", "")
        tag_name = re.sub(r"-+", "-", tag_name)
        tag_name = tag_name.strip("-")
        return tag_name

    def _save_checkpoint_files(self, args: TrainingArguments, ckpt_dir: str, metrics: dict = None, step: int = None):
        """Save model, trainer state files, and metrics.

        Note: This should only be called from rank 0 in the current implementation.
        """
        self._trainer.save_model(ckpt_dir)
        if args.should_save:
            self._trainer.save_state()  # trainer_state.json etc. in output_dir
            # save the trainer_state.json to the actual checkpoint directory
            shutil.copy(os.path.join(args.output_dir, "trainer_state.json"), ckpt_dir)

        # Save metrics to JSON file
        if metrics is not None:
            metrics_file = os.path.join(ckpt_dir, "metrics.json")
            metrics_to_save = {
                "step": step,
                "metrics": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in metrics.items()
                },
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            logger.info(f"📊 Saved metrics to {metrics_file}")

    def _cleanup_memory(self):
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _upload_checkpoint_to_hub(
        self, ckpt_dir: str, hub_model_id: str, tag_name: str, commit_message: str
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
        """
        Callback triggered after evaluation.
        Metrics are already gathered across all processes by the trainer before being passed here.
        """
        step = state.global_step

        # Only rank 0 needs to process metrics and save checkpoints
        if not self._is_main_process(self._trainer):
            logger.debug("Skipping checkpoint save (not main process)")
            return control

        logger.info(f"SaveBestCallback.on_evaluate called at step {step} with {len(metrics)} metrics")

        score, missing_metrics = self._compute_averaged_score(metrics)

        if missing_metrics:
            logger.warning(f"⚠️ Metrics {missing_metrics} not found in evaluation metrics")
            logger.warning(f"Available metrics: {metrics.keys()}")
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
            logger.info(
                f"💾 Saving ckpt: {ckpt_dir} | avg_score: {score:.6f} | {metrics_str} (rank {len(self._saved) + 1}/{self.keep_top_k})"
            )

            # Save model, trainer state, and metrics
            self._save_checkpoint_files(args, ckpt_dir, metrics, step)
            self._cleanup_memory()

            # Track that we saved a best checkpoint at this step
            self._last_best_save_step = step

            # Add to saved list and sort (always best -> worst since we normalized scores)
            self._saved.append((score, ckpt_dir))
            self._saved.sort(key=lambda x: x[0], reverse=True)

            # Remove old checkpoint if we exceed keep_top_k
            if len(self._saved) > self.keep_top_k:
                _, path_to_rm = self._saved.pop(-1)
                logger.info(f"🗑️ Removing old checkpoint: {path_to_rm}")
                if os.path.isdir(path_to_rm):
                    shutil.rmtree(path_to_rm, ignore_errors=True)

            # Upload to Hub if enabled and frequency check passes
            should_upload_to_hub = False
            if self.upload_to_hub:
                if self.hub_save_every is None:
                    # Upload every checkpoint if no frequency is set
                    should_upload_to_hub = True
                else:
                    # Check if it's time to upload based on frequency
                    if self._last_hub_upload_step == -1:
                        # First upload
                        should_upload_to_hub = True
                    elif (step - self._last_hub_upload_step) >= self.hub_save_every:
                        # Enough steps have passed
                        should_upload_to_hub = True

            if should_upload_to_hub:
                hub_model_id = self._get_hub_model_id(args)
                tag_name = self._clean_tag_name(f"best-{metric_short}-{score:.4f}-step-{step}")
                individual_scores_str = self._build_individual_scores_string(metrics)
                commit_message = f"Checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"

                logger.info(f"🚀 Uploading to Hub: {hub_model_id}")

                hub_url, commit_id = self._upload_checkpoint_to_hub(
                    ckpt_dir=ckpt_dir,
                    hub_model_id=hub_model_id,
                    tag_name=tag_name,
                    commit_message=commit_message,
                )
                logger.info(f"✅ Successfully uploaded to: {hub_url}")
                logger.info(f"🏷️ Tagged as: {tag_name}")

                # Track that we uploaded to Hub at this step
                self._last_hub_upload_step = step

                # Add to uploaded list and sort (always best -> worst since we normalized scores)
                self._uploaded.append((score, tag_name, commit_id))
                self._uploaded.sort(key=lambda x: x[0], reverse=True)

                # Remove old tags if we exceed keep_top_k
                api = HfApi(token=self.hub_token)
                if len(self._uploaded) > self.keep_top_k:
                    _, old_tag, _ = self._uploaded.pop(-1)
                    logger.info(f"🗑️ Removing old Hub tag: {old_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=old_tag)
                    logger.info(f"✅ Deleted tag: {old_tag}")

                # Aggressive memory cleanup after upload to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("🧹 Cleaned up memory after Hub upload")
            elif self.upload_to_hub and self.hub_save_every is not None:
                # Hub upload is enabled but not time yet
                steps_until_upload = self.hub_save_every - (step - self._last_hub_upload_step)
                logger.info(f"⏭️ Skipping Hub upload (saving locally only). Next upload in {steps_until_upload} steps")

        # Save 'latest' checkpoint if save_every is configured and it's time to save
        # Do this after processing best checkpoints so we have the gathered metrics
        # Skip if we just saved a best checkpoint at this step
        if self.save_every is not None and state.global_step > 0 and state.global_step % self.save_every == 0:
            if state.global_step != self._last_save_step and state.global_step != self._last_best_save_step:
                self._save_latest_checkpoint(args, state, metrics)
                self._last_save_step = state.global_step
            elif state.global_step == self._last_best_save_step:
                logger.info(
                    f"⏭️ Skipping 'latest' checkpoint save at step {state.global_step} (already saved as 'best')"
                )

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
        logger.info(f"💾 Saving 'latest' checkpoint at step {step} to {ckpt_dir} | {metrics_str}")

        # Remove old 'latest' checkpoint if it exists
        if self._previous_latest_ckpt_dir and os.path.isdir(self._previous_latest_ckpt_dir):
            logger.info(f"🗑️ Removing previous 'latest' checkpoint: {self._previous_latest_ckpt_dir}")
            shutil.rmtree(self._previous_latest_ckpt_dir, ignore_errors=True)

        # Save model, trainer state, and metrics
        self._save_checkpoint_files(args, ckpt_dir, metrics, step)
        logger.info(f"✅ Saved 'latest' checkpoint at step {step}")

        # Upload to Hub if enabled and frequency check passes
        should_upload_latest_to_hub = False
        if self.upload_to_hub:
            if self.hub_save_every is None:
                # Upload every checkpoint if no frequency is set
                should_upload_latest_to_hub = True
            else:
                # Check if it's time to upload based on frequency
                if self._last_latest_hub_upload_step == -1:
                    # First upload
                    should_upload_latest_to_hub = True
                elif (step - self._last_latest_hub_upload_step) >= self.hub_save_every:
                    # Enough steps have passed
                    should_upload_latest_to_hub = True

        if should_upload_latest_to_hub:
            hub_model_id = self._get_hub_model_id(args)
            tag_name = self._clean_tag_name(f"latest-{metric_short}-{score:.4f}-step-{step}")
            individual_scores_str = self._build_individual_scores_string(metrics)
            commit_message = f"Latest checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"

            # Delete previous 'latest' Hub tag if it exists
            api = HfApi(token=self.hub_token)
            if self._previous_latest_hub_tag:
                try:
                    logger.info(f"🗑️ Removing previous 'latest' Hub tag: {self._previous_latest_hub_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=self._previous_latest_hub_tag)
                    logger.info(f"✅ Deleted previous tag: {self._previous_latest_hub_tag}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete previous Hub tag {self._previous_latest_hub_tag}: {e}")

            logger.info(f"🚀 Uploading 'latest' checkpoint to Hub: {hub_model_id}")

            hub_url, commit_id = self._upload_checkpoint_to_hub(
                ckpt_dir=ckpt_dir,
                hub_model_id=hub_model_id,
                tag_name=tag_name,
                commit_message=commit_message,
            )
            logger.info(f"✅ Successfully uploaded 'latest' to: {hub_url}")
            logger.info(f"🏷️ Tagged as: {tag_name}")

            # Track this as the new previous latest
            self._previous_latest_hub_tag = tag_name
            self._last_latest_hub_upload_step = step
        elif self.upload_to_hub and self.hub_save_every is not None:
            # Hub upload is enabled but not time yet
            steps_until_upload = self.hub_save_every - (step - self._last_latest_hub_upload_step)
            logger.info(
                f"⏭️ Skipping Hub upload for 'latest' (saving locally only). Next upload in {steps_until_upload} steps"
            )

        # Track this as the new previous latest checkpoint directory
        self._previous_latest_ckpt_dir = ckpt_dir

        # Memory cleanup after saving model
        self._cleanup_memory()
