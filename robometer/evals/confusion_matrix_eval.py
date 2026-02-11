#!/usr/bin/env python3
"""
Standalone script to compute a combined confusion matrix across multiple eval datasets.

This script:
1. Loads N trajectories from each eval dataset
2. Creates confusion matrix samples (lang_task vs video_task) for each trajectory
3. Runs model inference on all samples
4. Builds a combined confusion matrix showing average final rewards

Usage:
    uv run python robometer/evals/confusion_matrix_eval.py \
        --n_trajectories_per_dataset 50 \
        --checkpoint_path rewardfm/rfm_qwen_pref_prog_4frames_all_strategy \
        --output_dir ./confusion_matrix_output \
        --batch_size 32
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from robometer.configs.experiment_configs import DataConfig, ExperimentConfig
from robometer.data.datasets.base import BaseDataset, resolve_dataset_keys
from robometer.data.datasets.helpers import create_trajectory_from_dict
from robometer.data.samplers.base import RFMBaseSampler
from robometer.data.dataset_types import ProgressSample
from robometer.evals.eval_server import forward_model
from robometer.models.utils import convert_bins_to_continuous
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

# Default eval datasets for confusion matrix computation
EVAL_DATASETS = [
    # "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm",
    # "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
    # "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
    # "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
    # "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top"
]

logger = get_logger()


ALLOWED_TASKS = [
    "Move the orange cup from left to right",
    "Open the red trash bin",
    "Separate the purple and red cups",
    "Throw the black marker away in the blue trash can",
]


def sample_trajectories_from_dataset(dataset: BaseDataset, n_trajectories: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample N successful trajectories from a dataset.

    Args:
        dataset: BaseDataset instance
        n_trajectories: Number of trajectories to sample (only from successful trajectories)
        seed: Random seed for reproducibility

    Returns:
        List of trajectory dictionaries (all successful)
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset_size = len(dataset.dataset)

    # First, filter for successful trajectories only
    successful_trajectories = []
    successful_indices = []
    for idx in range(dataset_size):
        traj = dataset.dataset[idx]
        quality_label = traj.get("quality_label")
        if quality_label == "successful" and traj.get("task") in ALLOWED_TASKS:
            successful_trajectories.append(traj)
            successful_indices.append(idx)

    num_successful = len(successful_trajectories)
    logger.info(f"Found {num_successful} successful trajectories out of {dataset_size} total trajectories")

    if num_successful == 0:
        logger.warning("No successful trajectories found in dataset")
        return []

    # Sample N from successful trajectories
    if n_trajectories is None or n_trajectories > num_successful:
        n_trajectories = num_successful

    # Sample random indices from successful trajectories
    sampled_indices = random.sample(range(num_successful), n_trajectories)
    trajectories = [successful_trajectories[idx] for idx in sampled_indices]

    logger.info(f"Sampled {len(trajectories)} successful trajectories from {num_successful} available")
    return trajectories


def get_all_unique_tasks(trajectories_by_dataset: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Get all unique tasks across all datasets.

    Args:
        trajectories_by_dataset: Dictionary mapping dataset name to list of trajectories

    Returns:
        Sorted list of unique task names
    """
    all_tasks = set()
    for dataset_name, trajectories in trajectories_by_dataset.items():
        for traj in trajectories:
            task = traj.get("task")
            if task is not None:
                all_tasks.add(task)

    unique_tasks = sorted(list(all_tasks))
    logger.info(f"Found {len(unique_tasks)} unique tasks across all datasets")
    return unique_tasks


def create_confusion_matrix_samples(
    trajectories_by_dataset: Dict[str, List[Dict[str, Any]]],
    all_lang_tasks: List[str],
    dataset_by_name: Dict[str, BaseDataset],
    data_config: DataConfig,
) -> List[ProgressSample]:
    """Create confusion matrix samples for all trajectory-lang_task pairs.

    For each trajectory, creates samples with each lang_task as the instruction.

    Args:
        trajectories_by_dataset: Dictionary mapping dataset name to list of trajectories
        all_lang_tasks: List of all unique tasks to use as language instructions
        dataset_by_name: Dictionary mapping dataset name to BaseDataset instance
        data_config: DataConfig for creating sampler

    Returns:
        List of ProgressSample objects for confusion matrix evaluation
    """
    samples = []

    for dataset_name, trajectories in tqdm(trajectories_by_dataset.items(), desc="Creating confusion matrix samples"):
        # Get the dataset and create a sampler instance for this dataset
        base_dataset = dataset_by_name[dataset_name]

        # Create a minimal sampler instance to use _get_traj_from_data
        sampler = RFMBaseSampler(
            config=data_config,
            dataset=base_dataset.dataset,
            combined_indices=base_dataset._combined_indices,
            dataset_success_cutoff_map=base_dataset.dataset_success_cutoff_map,
            verbose=False,
        )

        for traj in trajectories:
            video_task = traj.get("task")
            if video_task is None:
                continue

            # Create a sample for each lang_task
            for lang_task in all_lang_tasks:
                # Create trajectory dict with lang_task as the instruction
                traj_dict = traj.copy()
                traj_dict["task"] = lang_task  # Override task with lang_task for instruction
                # Remove text_embedding if present (will be recomputed by collator if needed)
                # This ensures we use the correct embedding for the lang_task, not the original task
                if "text_embedding" in traj_dict:
                    del traj_dict["text_embedding"]

                # Create metadata with both lang_task and video_task
                metadata = {
                    "id": traj.get("id"),
                    "lang_task": lang_task,
                    "video_task": video_task,
                    "video_path": traj.get("frames") if isinstance(traj.get("frames"), str) else None,
                }

                # Use sampler's _get_traj_from_data to properly load and process trajectory
                trajectory = sampler._get_traj_from_data(
                    traj=traj_dict,
                    metadata=metadata,
                )

                if trajectory is None:
                    logger.warning(f"Failed to create trajectory for traj {traj.get('id')} with lang_task {lang_task}")
                    continue

                # Create ProgressSample
                sample = ProgressSample(trajectory=trajectory)
                samples.append(sample)

    logger.info(f"Created {len(samples)} confusion matrix samples")
    return samples


def process_samples_with_model(
    samples: List[ProgressSample],
    model: Any,
    batch_collator: Any,
    batch_size: int = 32,
    device: torch.device = None,
) -> List[Dict[str, Any]]:
    """Process confusion matrix samples with the model.

    Args:
        samples: List of ProgressSample objects
        model: RFM/ReWiND model
        batch_collator: Batch collator for processing
        batch_size: Batch size for processing
        device: Device to run inference on

    Returns:
        List of result dictionaries with progress predictions and metadata
    """
    results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch = samples[batch_start : batch_start + batch_size]

        # Collate batch
        batch_inputs = batch_collator(batch)
        progress_inputs = batch_inputs["progress_inputs"]

        # Move to device
        if device is not None:
            progress_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in progress_inputs.items()
            }

        # Forward pass
        with torch.inference_mode():
            model_output, _ = forward_model(model, progress_inputs, sample_type="progress")

        # Extract progress logits
        progress_logits = model_output.progress_logits
        if progress_logits is None:
            logger.warning("No progress logits returned from model")
            continue

        # Handle different output formats
        if isinstance(progress_logits, dict):
            progress_tensor = progress_logits.get("A")
        else:
            progress_tensor = progress_logits

        if progress_tensor is None:
            logger.warning("No progress logits in 'A' key")
            continue

        # Process each sample in the batch
        for i, sample in enumerate(batch):
            if i >= progress_tensor.shape[0]:
                continue

            # Extract progress prediction for this sample (full sequence)
            # Convert to float32 first to handle bfloat16 tensors (NumPy doesn't support bfloat16)
            pred_array = progress_tensor[i].detach().cpu().float().numpy()

            # Store full sequence of predictions (confusion matrix eval extracts final reward)
            if pred_array.ndim > 1:
                # Multi-dimensional: could be [seq_len, num_bins] for discrete or [seq_len] for continuous
                progress_pred = pred_array.tolist()
            else:
                # Single dimension: convert to list
                progress_pred = pred_array.tolist() if pred_array.ndim > 0 else [float(pred_array)]

            # Get metadata from trajectory
            metadata = sample.trajectory.metadata or {}
            lang_task = metadata.get("lang_task")
            video_task = metadata.get("video_task")

            result = {
                "progress_pred": progress_pred,
                "metadata": metadata,
                "task": lang_task,  # Use lang_task as task for consistency with confusion matrix eval
            }
            results.append(result)

    logger.info(f"Processed {len(results)} samples")
    return results


def compute_combined_confusion_matrix(
    results: List[Dict[str, Any]],
    output_dir: str,
    progress_pred_type: str = "absolute",
    is_discrete_mode: bool = False,
    num_bins: int = 10,
) -> None:
    """Compute and save the combined confusion matrix.

    Args:
        results: List of result dictionaries from model inference
        output_dir: Output directory for saving results
        progress_pred_type: Progress prediction type ("absolute" or "relative")
        is_discrete_mode: Whether model uses discrete progress prediction
        num_bins: Number of bins for discrete mode
    """
    # First, gather all progress predictions, lang_tasks, and video_tasks
    all_progress_preds = []
    all_lang_tasks = []
    all_video_tasks = []
    valid_indices = []

    for idx, r in enumerate(results):
        progress_pred = r.get("progress_pred")
        if progress_pred is not None and len(progress_pred) > 0:
            meta = r.get("metadata", {})
            lang_task = meta.get("lang_task")
            video_task = meta.get("video_task")
            if lang_task is not None and video_task is not None:
                all_progress_preds.append(progress_pred)
                all_lang_tasks.append(lang_task)
                all_video_tasks.append(video_task)
                valid_indices.append(idx)

    if not all_progress_preds:
        logger.error("No valid confusion matrix data found")
        return

    # Group results by confusion matrix task
    uniq_tasks = set(all_lang_tasks) | set(all_video_tasks)
    task_to_idx = {task: idx for idx, task in enumerate(sorted(uniq_tasks))}
    num_tasks = len(uniq_tasks)

    # Extract final rewards vectorized
    all_final_rewards = []
    for progress_pred in all_progress_preds:
        pred_array = np.array(progress_pred)

        if is_discrete_mode:
            # Discrete mode: progress_pred is logits [seq_len, num_bins]
            # Convert to continuous values using weighted sum of bin centers
            last_frame_logits = pred_array[-1] if pred_array.ndim > 1 else pred_array
            continuous_pred = convert_bins_to_continuous(torch.tensor(last_frame_logits, dtype=torch.float32)).item()
            final_reward = float(continuous_pred)
        else:
            # Continuous mode: use last frame value
            if progress_pred_type == "relative":
                pred_array = np.cumsum(pred_array)
            final_reward = float(pred_array[-1] if pred_array.ndim > 0 else pred_array)

        all_final_rewards.append(final_reward)

    all_final_rewards = np.array(all_final_rewards)
    all_lang_indices = np.array([task_to_idx[task] for task in all_lang_tasks])
    all_video_indices = np.array([task_to_idx[task] for task in all_video_tasks])

    # Build confusion matrix using vectorized operations
    confusion_matrix = np.zeros((num_tasks, num_tasks))
    count_matrix = np.zeros((num_tasks, num_tasks))

    # Use advanced indexing to accumulate rewards
    np.add.at(confusion_matrix, (all_lang_indices, all_video_indices), all_final_rewards)
    np.add.at(count_matrix, (all_lang_indices, all_video_indices), 1)

    if np.sum(count_matrix) == 0:
        logger.error("No valid confusion matrix data found")
        return

    # Calculate average rewards (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        confusion_matrix = np.divide(
            confusion_matrix, count_matrix, out=np.zeros_like(confusion_matrix), where=count_matrix != 0
        )

    # Create the plot
    fig = plt.figure(figsize=(10, 8))

    # Create heatmap showing average final rewards
    sns.heatmap(
        confusion_matrix,
        cmap="Blues",  # White to dark blue colormap
    )
    plt.xlabel("Language Task", fontsize=12)
    plt.ylabel("Video Task", fontsize=12)
    # Remove xticks and yticks
    plt.xticks([])
    plt.yticks([])

    # Remove the legend
    plt.legend([])

    plt.tight_layout()

    # Save confusion matrix figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved confusion matrix figure to {fig_path}")

    # Save confusion matrix as numpy array
    matrix_path = os.path.join(output_dir, "confusion_matrix.npy")
    np.save(matrix_path, confusion_matrix)
    logger.info(f"Saved confusion matrix array to {matrix_path}")

    # Save confusion matrix as JSON (with task labels)
    # Extract unique tasks from results
    all_lang_tasks = sorted(
        list(set(r.get("metadata", {}).get("lang_task") for r in results if r.get("metadata", {}).get("lang_task")))
    )
    all_video_tasks = sorted(
        list(set(r.get("metadata", {}).get("video_task") for r in results if r.get("metadata", {}).get("video_task")))
    )

    # Create task mapping (same order as in confusion matrix)
    all_unique_tasks = sorted(list(set(all_lang_tasks) | set(all_video_tasks)))
    task_to_idx = {task: idx for idx, task in enumerate(all_unique_tasks)}

    json_data = {
        "confusion_matrix": confusion_matrix.tolist(),
        "task_labels": all_unique_tasks,
        "task_to_idx": task_to_idx,
        "lang_tasks": all_lang_tasks,
        "video_tasks": all_video_tasks,
    }
    json_path = os.path.join(output_dir, "confusion_matrix.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved confusion matrix JSON to {json_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute combined confusion matrix across multiple eval datasets")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=EVAL_DATASETS,
        help=f"List of eval dataset names to load trajectories from (default: {EVAL_DATASETS})",
    )
    parser.add_argument(
        "--n_trajectories_per_dataset",
        type=int,
        default=30,
        help="Number of trajectories to sample from each dataset (default: 10)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (HuggingFace repo ID or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for saving confusion matrix results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing samples (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling trajectories (default: 42)",
    )
    parser.add_argument(
        "--progress_pred_type",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help="Progress prediction type (default: absolute)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_loguru_logging(args.log_level)

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    exp_config, tokenizer, processor, model = load_model_from_hf(
        model_path=args.checkpoint_path,
        device=device,
    )
    model.eval()

    # Create batch collator
    batch_collator = setup_batch_collator(
        processor=processor,
        tokenizer=tokenizer,
        cfg=exp_config,
        is_eval=True,
    )

    # Determine if discrete mode
    is_discrete_mode = exp_config.loss.progress_loss_type.lower() == "discrete"
    num_bins = exp_config.loss.progress_discrete_bins if is_discrete_mode else None
    logger.info(f"Progress prediction mode: {'discrete' if is_discrete_mode else 'continuous'} (bins: {num_bins})")

    # Check if load_embeddings is enabled (not recommended for confusion matrix since we override tasks)
    if exp_config.data.load_embeddings:
        logger.warning(
            "load_embeddings=True is enabled. This may cause issues when overriding tasks for confusion matrix. "
            "Consider setting load_embeddings=False for confusion matrix evaluation."
        )

    # Resolve dataset keys
    resolved_datasets = resolve_dataset_keys(args.eval_datasets, split="eval")
    logger.info(f"Resolved eval datasets: {resolved_datasets}")

    # Load trajectories from each dataset
    trajectories_by_dataset = {}
    dataset_by_name = {}
    for dataset_name in resolved_datasets:
        logger.info(f"Loading trajectories from dataset: {dataset_name}")

        # Create data config for this dataset
        data_cfg = DataConfig(
            eval_datasets=[dataset_name],
            max_frames=exp_config.data.max_frames,
            progress_pred_type=exp_config.data.progress_pred_type,
            load_embeddings=exp_config.data.load_embeddings,
            max_success=exp_config.data.max_success,
            progress_loss_type=exp_config.loss.progress_loss_type,
            progress_discrete_bins=exp_config.loss.progress_discrete_bins if is_discrete_mode else None,
        )

        # Load base dataset
        base_dataset = BaseDataset(config=data_cfg, is_evaluation=True)
        dataset_by_name[dataset_name] = base_dataset

        # Sample N trajectories
        trajectories = sample_trajectories_from_dataset(
            base_dataset,
            n_trajectories=args.n_trajectories_per_dataset,
            seed=args.seed,
        )
        trajectories_by_dataset[dataset_name] = trajectories

        logger.info(f"Loaded {len(trajectories)} trajectories from {dataset_name}")

    # Get all unique tasks across all datasets (these will be lang_tasks)
    all_lang_tasks = get_all_unique_tasks(trajectories_by_dataset)
    logger.info(f"Unique tasks to use as language instructions: {all_lang_tasks}")

    # Create confusion matrix samples
    logger.info("Creating confusion matrix samples...")
    # Create a data config for the sampler (use first dataset's config as template)
    sampler_data_cfg = DataConfig(
        max_frames=exp_config.data.max_frames,
        progress_pred_type=exp_config.data.progress_pred_type,
        load_embeddings=exp_config.data.load_embeddings,
        max_success=exp_config.data.max_success,
        progress_loss_type=exp_config.loss.progress_loss_type,
        progress_discrete_bins=exp_config.loss.progress_discrete_bins if is_discrete_mode else None,
    )
    samples = create_confusion_matrix_samples(
        trajectories_by_dataset,
        all_lang_tasks,
        dataset_by_name,
        sampler_data_cfg,
    )

    # Process samples with model
    logger.info("Processing samples with model...")
    results = process_samples_with_model(
        samples=samples,
        model=model,
        batch_collator=batch_collator,
        batch_size=args.batch_size,
        device=device,
    )

    # Compute and save confusion matrix
    logger.info("Computing confusion matrix...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw results
    results_path = os.path.join(args.output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved raw results to {results_path}")

    # Compute confusion matrix
    compute_combined_confusion_matrix(
        results=results,
        output_dir=args.output_dir,
        progress_pred_type=args.progress_pred_type,
        is_discrete_mode=is_discrete_mode,
        num_bins=num_bins if num_bins else 10,
    )

    logger.info(f"Confusion matrix computation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
