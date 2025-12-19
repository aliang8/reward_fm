#!/usr/bin/env python3
"""
Script to compile evaluation results from JSON files.
"""

import argparse
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any
import yaml
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from rfm.data.datasets.helpers import load_frames_from_npz
from rfm.data.dataset_category import is_failure
from rfm.evals.eval_metrics_utils import compute_pearson, compute_preference_accuracy, compute_spearman


def compute_eval_metrics(eval_type: str, results: list[dict[str, Any]], progress_pred_type: str):
    if eval_type == "quality_preference" or eval_type == "quality_preference_roboarena":
        return run_quality_preference_eval(results, progress_pred_type)
    elif eval_type == "reward_alignment":
        return run_reward_alignment_eval_per_trajectory(results, progress_pred_type)
    elif eval_type == "confusion_matrix":
        return run_confusion_matrix_eval(results, progress_pred_type)
    elif eval_type == "policy_ranking":
        return run_policy_ranking_eval(results, progress_pred_type)
    elif eval_type == "similarity_score":
        return run_similarity_score_eval(results, progress_pred_type)


def run_quality_preference_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run quality_preference evaluation analysis.

    Groups results by task and quality labels (or partial_success for RoboArena),
    computes preference accuracy per group and aggregate.
    Returns metrics, task_groups, and task_details similar to policy_ranking.
    """

    # First, gather all logits and labels, convert to arrays
    all_preds = []
    all_labels = []
    all_tasks = []
    all_quality_combos = []
    valid_indices = []
    use_partial_success = False

    # Check if we should use partial_success (RoboArena) or quality_label
    if results and len(results) > 0:
        first_r = results[0]
        chosen_meta = first_r.get("metadata", {}).get("chosen_metadata", {})
        if chosen_meta and "partial_success" in chosen_meta:
            use_partial_success = True

    for idx, r in enumerate(results):
        pred = r.get("preference_pred")
        label = r.get("preference_labels")
        if pred is not None and label is not None:
            if isinstance(pred, np.ndarray):
                pred = float(pred.item()) if pred.size == 1 else float(pred[0])
            else:
                pred = float(pred)

            if isinstance(label, np.ndarray):
                label = float(label.item()) if label.size == 1 else float(label[0])
            else:
                label = float(label)

            # For non-RoboArena, extract quality combo; for RoboArena, just validate metadata exists
            chosen_meta = r.get("metadata", {}).get("chosen_metadata", {})
            rejected_meta = r.get("metadata", {}).get("rejected_metadata", {})

            if use_partial_success:
                # For RoboArena, just check that partial_success exists (we don't use it)
                chosen_val = chosen_meta.get("partial_success")
                rejected_val = rejected_meta.get("partial_success")
                if chosen_val is None or rejected_val is None:
                    continue
            else:
                # For non-RoboArena, extract quality combo for later use
                chosen_val = chosen_meta.get("quality_label")
                rejected_val = rejected_meta.get("quality_label")
                if chosen_val is None or rejected_val is None:
                    continue
                combo_key = tuple(sorted([chosen_val, rejected_val]))
                all_quality_combos.append(combo_key)

            all_preds.append(pred)
            all_labels.append(label)
            all_tasks.append(r["task"])
            valid_indices.append(idx)

    if not all_preds:
        return {"error": "No valid predictions found"}, {}, {}

    # Convert to numpy arrays for vectorized operations
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Convert logits to binary predictions (1 if pred > 0, else 0)
    binary_preds = (all_preds > 0).astype(float)

    # Group results by task (using valid indices to map back)
    task_groups = defaultdict(list)
    task_indices = defaultdict(list)
    for i, (idx, task) in enumerate(zip(valid_indices, all_tasks)):
        task_groups[task].append(results[idx])
        task_indices[task].append(i)

    # Compute preference accuracy per task group using vectorized operations
    task_details = {}
    all_correct = 0
    all_total = 0

    for task, task_results in task_groups.items():
        task_idx = task_indices[task]
        task_preds = binary_preds[task_idx]
        task_labels = all_labels[task_idx]

        task_correct = np.sum(task_preds == task_labels)
        task_total = len(task_preds)
        pref_acc = task_correct / task_total if task_total > 0 else 0.0

        task_detail = {
            "preference_accuracy": pref_acc,
            "num_correct": int(task_correct),
            "num_total": task_total,
        }

        # Only compute quality accuracies for non-RoboArena datasets
        if not use_partial_success:
            # Compute accuracy per quality combination using vectorized operations
            task_quality_combos = [all_quality_combos[i] for i in task_idx]
            quality_accs = {}
            unique_combos = set(task_quality_combos)
            for combo_key in unique_combos:
                combo_mask = np.array([qc == combo_key for qc in task_quality_combos])
                combo_preds = task_preds[combo_mask]
                combo_labels = task_labels[combo_mask]
                if len(combo_preds) > 0:
                    combo_correct = np.sum(combo_preds == combo_labels)
                    combo_acc = combo_correct / len(combo_preds) if len(combo_preds) > 0 else 0.0
                    quality_accs[f"{combo_key[0]}_vs_{combo_key[1]}"] = combo_acc
            task_detail["quality_accuracies"] = quality_accs
        else:
            task_detail["quality_accuracies"] = None

        task_details[task] = task_detail

        all_correct += task_correct
        all_total += task_total

    # Aggregate metrics
    aggregate_acc = all_correct / all_total if all_total > 0 else 0.0

    metrics = {
        "preference_accuracy": aggregate_acc,
    }

    return metrics, task_groups, task_details


def run_reward_alignment_eval_per_trajectory(
    results: list[dict[str, Any]], progress_pred_type: str, last_frame_only: bool = False
) -> tuple[dict[str, Any], list, list, list]:
    """Run reward_alignment evaluation analysis and create plots for each trajectory.

    For failure datasets, we visualize predictions but skip metric computation.

    Returns:
        Tuple of (metrics, plots, video_frames_list, trajectory_progress_data)
        where trajectory_progress_data is a list of dicts with progress_pred and target_progress
        for each trajectory (one per video in video_frames_list)
    """
    # Determine if this is a failure dataset by checking the data_source of the first result
    is_failure_dataset = False
    if results and len(results) > 0:
        first_data_source = results[0].get("data_source", "")
        is_failure_dataset = is_failure(first_data_source)

    unique_trajectory_ids = set()
    loss_per_trajectory = np.zeros(1)
    loss_trajectories = []
    spearman_trajectories = []
    plots = []
    video_frames_list = []
    trajectory_progress_data = []
    
    # Detect if we're using discrete predictions by checking the first result
    is_discrete_mode = False
    if results and len(results) > 0:
        first_pred = results[0].get("progress_pred")
        if first_pred is not None:
            pred_array = np.array(first_pred)
            # Check if it's logits: should have shape [seq_len, num_bins] or [num_bins]
            if pred_array.ndim >= 1 and pred_array.shape[-1] > 1:
                num_bins = int(pred_array.shape[-1])
                is_discrete_mode = True

    # Collect all success_probs and success_labels for AUPRC computation
    all_success_probs = []
    all_success_labels = []

    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)

        # Collect success probabilities and labels for AUPRC
        success_probs = r.get("success_probs", None)
        success_labels = r.get("success_labels", None)
        if success_probs is not None and success_labels is not None:
            # Convert to numpy arrays if needed
            if isinstance(success_probs, np.ndarray):
                all_success_probs.append(success_probs.flatten())
            else:
                all_success_probs.append(np.array(success_probs).flatten())

            if isinstance(success_labels, np.ndarray):
                all_success_labels.append(success_labels.flatten())
            else:
                all_success_labels.append(np.array(success_labels).flatten())

    for trajectory_id in unique_trajectory_ids:
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]
        results_for_trajectory.sort(key=lambda r: r["metadata"]["subsequence_end"])

        # Get task and quality label from first result
        task = results_for_trajectory[0].get("task", "unknown")
        quality_label = results_for_trajectory[0].get("quality_label", "unknown")

        # For failure datasets, visualize all trajectories; otherwise only successful ones
        if not is_failure_dataset and quality_label != "successful":
            continue

        # Try to get video_path from results, if not available, we'll return None for frames
        video_path = results_for_trajectory[0].get("video_path", None)

        # Determine success availability from the first result only
        have_success = results_for_trajectory[0].get("success_pred", None) is not None
        have_success_labels = results_for_trajectory[0].get("success_labels", None) is not None

        # First, gather all predictions and targets for this trajectory
        all_preds = []
        all_targets = []
        all_pred_logits = []  # For discrete mode: collect full logits
        all_target_bins = []  # For discrete mode: collect bin indices
        all_success_preds = []
        all_success_labels_list = []

        for timestep, r in enumerate(results_for_trajectory):
            pred = r.get("progress_pred")
            tgt = r.get("target_progress")

            if pred is not None:
                pred_array = np.array(pred)
                if is_discrete_mode:
                    # Discrete mode: pred is logits [seq_len, num_bins]
                    if last_frame_only:
                        # Use last frame's logits
                        all_pred_logits.append(pred_array[-1])
                        if tgt is not None and len(tgt) > 0:
                            all_target_bins.append(int(tgt[-1]))
                    else:
                        # Use prediction at current timestep
                        if timestep >= len(pred_array) - 1:
                            indx = -1
                        else:
                            indx = timestep
                        all_pred_logits.append(pred_array[indx])
                        if tgt is not None and len(tgt) > 0:
                            # Target is already a discrete bin index
                            if timestep >= len(tgt) - 1:
                                all_target_bins.append(int(tgt[-1]))
                            else:
                                all_target_bins.append(int(tgt[indx]))
                    # For visualization: use argmax to get predicted bin (raw integer)
                    if last_frame_only:
                        pred_bin = np.argmax(pred_array[-1])
                    else:
                        if timestep >= len(pred_array) - 1:
                            pred_bin = np.argmax(pred_array[-1])
                        else:
                            pred_bin = np.argmax(pred_array[timestep])
                    all_preds.append(int(pred_bin))
                else:
                    # Continuous mode: original logic
                    if last_frame_only:
                        all_preds.append(float(pred[-1]))
                    else:
                        if timestep >= len(pred) - 1:
                            indx = -1
                        else:
                            indx = timestep
                        all_preds.append(float(pred[indx]))
            else:
                all_preds.append(0.0)

            if tgt is not None and len(tgt) > 0:
                if is_discrete_mode:
                    # Target is already a discrete bin index (raw integer)
                    tgt_bin = int(tgt[-1] if last_frame_only else (tgt[-1] if timestep >= len(tgt) - 1 else tgt[timestep]))
                    all_targets.append(tgt_bin)
                else:
                    all_targets.append(float(tgt[-1]))
            else:
                all_targets.append(0 if is_discrete_mode else 0.0)

            # Optional success prediction (binary) from trainer outputs
            succ = r.get("success_pred", None)
            if succ is not None and len(succ) > 0:
                all_success_preds.append(float(succ[-1]))

            # Optional success labels (ground truth) from trainer outputs
            succ_labels = r.get("success_labels", None)
            if succ_labels is not None and len(succ_labels) > 0:
                all_success_labels_list.append(float(succ_labels[-1]))

        if len(all_preds) == 0 or len(all_targets) == 0:
            print("No valid predictions or targets found for trajectory: ", trajectory_id)
            continue

        last_preds = np.array(all_preds)
        last_targets = np.array(all_targets)
        last_success = np.array(all_success_preds) if all_success_preds else None
        last_success_labels = np.array(all_success_labels_list) if all_success_labels_list else None

        # Load video frames if video path exists
        frames = None
        if video_path:
            try:
                frames = load_frames_from_npz(video_path)
                frames = frames.transpose(0, 3, 1, 2)

                # Resize frames to make them smaller for wandb table display
                resized_frames = []
                for frame in frames:
                    frame_resized = cv2.resize(frame.transpose(1, 2, 0), (64, 64))
                    resized_frames.append(frame_resized.transpose(2, 0, 1))
                frames = np.array(resized_frames)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                frames = None

        video_frames_list.append(frames)

        if progress_pred_type == "relative":
            last_preds = np.cumsum(last_preds)
            last_targets = np.cumsum(last_targets)

        trajectory_progress_data.append({
            "progress_pred": last_preds.tolist() if isinstance(last_preds, np.ndarray) else last_preds,
            "target_progress": last_targets.tolist() if isinstance(last_targets, np.ndarray) else last_targets,
        })

        # Calculate metrics for this trajectory using vectorized operations (skip for failure datasets)
        if is_failure_dataset:
            traj_loss = 0.0
            traj_spearman = 0.0
        else:
            # Compute loss based on mode
            if is_discrete_mode and all_pred_logits and all_target_bins:
                # Discrete mode: compute cross-entropy loss between logits and target bins
                pred_logits_tensor = torch.tensor(np.array(all_pred_logits), dtype=torch.float32)  # [seq_len, num_bins]
                target_bins_tensor = torch.tensor(all_target_bins, dtype=torch.long)  # [seq_len]
                target_bins_tensor = torch.clamp(target_bins_tensor, 0, num_bins - 1)
                loss_per_timestep = F.cross_entropy(pred_logits_tensor, target_bins_tensor, reduction="none")
                traj_loss = float(loss_per_timestep.mean().item())
            else:
                # Continuous mode: compute MSE loss
                traj_loss = float(np.mean((last_targets - last_preds) ** 2))
            
            # Compute Spearman correlation (shared for both modes)
            traj_spearman = compute_spearman(last_targets.tolist(), last_preds.tolist())
            # Handle NaN values
            traj_spearman = float(traj_spearman) if not np.isnan(traj_spearman) else 0.0

        # Create a wandb plot for progress predictions and, if available, success predictions
        if have_success and last_success is not None and len(last_success) == len(last_preds):
            fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
            # Progress subplot
            ax = axs[0]
            # Success subplot (binary)
            ax2 = axs[1]
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax2 = None
        
        # Setup progress plot (shared code for both cases)
        ax.plot(last_preds, linewidth=2)
        ax.set_ylabel("Progress")
        title = f"Progress - {task} - {quality_label}\nLoss: {traj_loss:.3f}, sp: {traj_spearman:.2f}"
        ax.set_ylim(0, num_bins - 1 if is_discrete_mode else 1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_yticks([])
        ax.set_title(title)

        # Setup success subplot if available
        if ax2 is not None:
            ax2.step(range(len(last_success)), last_success, where="post", linewidth=2, label="Predicted", color="blue")
            # Add ground truth success labels as green line if available
            if (
                have_success_labels
                and last_success_labels is not None
                and len(last_success_labels) == len(last_success)
            ):
                ax2.step(
                    range(len(last_success_labels)),
                    last_success_labels,
                    where="post",
                    linewidth=2,
                    label="Ground Truth",
                    color="green",
                )
            ax2.set_ylabel("Success")
            ax2.set_title("Success (final per slice)")
            ax2.set_ylim(-0.05, 1.05)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.set_yticks([0, 1])
            if have_success_labels and len(last_success_labels) == len(last_success):
                ax2.legend(loc="upper right", fontsize=8)

        plots.append(fig)

        # Only accumulate metrics for non-failure datasets (using already computed values)
        if not is_failure_dataset:
            loss_trajectories.append(traj_loss)
            if not np.isnan(traj_spearman):
                spearman_trajectories.append(traj_spearman)

    if len(unique_trajectory_ids) == 0:
        loss_per_trajectory = np.nan
        spearman_per_trajectory = np.nan
    else:
        loss_per_trajectory = np.mean(loss_trajectories).item() if loss_trajectories else np.nan
        spearman_per_trajectory = np.mean(spearman_trajectories).item() if spearman_trajectories else np.nan

    # Compute success_auprc across all collected success predictions and labels
    success_auprc = None
    if all_success_probs and all_success_labels:
        # Flatten all collected probabilities and labels
        success_probs_flat = np.concatenate(all_success_probs)
        success_labels_flat = np.concatenate(all_success_labels)

        # Compute AUPRC if we have valid data
        if success_probs_flat.size > 0 and len(np.unique(success_labels_flat)) > 1:
            success_auprc = float(average_precision_score(success_labels_flat, success_probs_flat))
        else:
            success_auprc = 0.0

    metrics = {
        "loss": loss_per_trajectory,
        "spearman": spearman_per_trajectory,
    }

    # Add success_auprc to metrics if computed
    if success_auprc is not None:
        metrics["success_auprc"] = success_auprc

    return metrics, plots, video_frames_list, trajectory_progress_data


def run_confusion_matrix_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run confusion_matrix evaluation analysis."""
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
        return {"error": "No valid confusion matrix data found"}, np.zeros((1, 1))

    # Group results by confusion matrix task
    uniq_tasks = set(all_lang_tasks) | set(all_video_tasks)
    task_to_idx = {task: idx for idx, task in enumerate(uniq_tasks)}
    num_tasks = len(uniq_tasks)

    # Extract final rewards vectorized
    all_final_rewards = []
    for progress_pred in all_progress_preds:
        if progress_pred_type == "relative":
            progress_pred = np.cumsum(progress_pred)
        all_final_rewards.append(float(progress_pred[-1]))

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
        return {"error": "No valid confusion matrix data found"}, np.zeros((num_tasks, num_tasks))

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
        # annot=True,
        # fmt='.3f',
        cmap="Blues",  # White to dark blue colormap
        # xticklabels=list(uniq_tasks),
        # yticklabels=list(uniq_tasks),
        # cbar_kws={"label": "Average Final Reward (5 trajs)"},
    )
    plt.xlabel("Language Task", fontsize=12)
    plt.ylabel("Video Task", fontsize=12)
    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=0)
    # Remove xticks and yticks
    plt.xticks([])
    plt.yticks([])

    # Remove the legend
    plt.legend([])

    plt.tight_layout()

    return fig, confusion_matrix


def run_policy_ranking_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run policy_ranking evaluation analysis.

    For non-RoboArena: Uses quality_label and quality_order for ranking.
    For RoboArena: Uses partial_success for ranking (no quality_order computation).
    """
    # Check if this is RoboArena (uses partial_success instead of quality_label)
    use_partial_success = False
    if results and len(results) > 0:
        first_r = results[0]
        if first_r.get("partial_success") is not None:
            use_partial_success = True

    # First, gather all progress predictions, tasks, quality_labels/partial_success
    all_progress_preds = []
    all_tasks = []
    all_quality_labels = []
    all_partial_successes = []
    all_video_paths = []
    all_ids = []
    valid_indices = []

    for idx, r in enumerate(results):
        progress_pred = r.get("progress_pred")
        if progress_pred is not None and len(progress_pred) > 0:
            task = r.get("task")
            if task is None:
                continue

            if use_partial_success:
                partial_success = r.get("partial_success")
                if partial_success is not None:
                    all_progress_preds.append(progress_pred)
                    all_tasks.append(task)
                    all_partial_successes.append(float(partial_success))
                    all_video_paths.append(r.get("video_path"))
                    all_ids.append(r.get("id"))
                    valid_indices.append(idx)
            else:
                quality_label = r.get("quality_label")
                if quality_label is not None:
                    all_progress_preds.append(progress_pred)
                    all_tasks.append(task)
                    all_quality_labels.append(quality_label)
                    all_video_paths.append(r.get("video_path"))
                    valid_indices.append(idx)

    if not all_progress_preds:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Extract final rewards vectorized
    all_final_rewards = []
    for progress_pred in all_progress_preds:
        if progress_pred_type == "relative":
            progress_pred = np.cumsum(progress_pred)
        all_final_rewards.append(float(progress_pred[-1]))

    all_final_rewards = np.array(all_final_rewards)
    if use_partial_success:
        all_partial_successes = np.array(all_partial_successes)

    # Group by task
    task_groups = {}
    task_indices = defaultdict(list)
    for i, (idx, task) in enumerate(zip(valid_indices, all_tasks)):
        if task not in task_groups:
            task_groups[task] = []
        if use_partial_success:
            task_groups[task].append({
                "partial_success": all_partial_successes[i],
                "final_predicted_reward": all_final_rewards[i],
                "video_path": all_video_paths[i],
                "id": all_ids[i],
            })
        else:
            task_groups[task].append({
                "quality_label": all_quality_labels[i],
                "final_reward": all_final_rewards[i],
                "video_path": all_video_paths[i],
            })
        task_indices[task].append(i)

    if not task_groups:
        return {"error": "No valid policy ranking data found"}, {}, {}

    task_details = {}
    all_spearman = []
    all_spearman_rewind = []
    all_succ_subopt_diffs = []
    all_subopt_fail_diffs = []
    all_succ_fail_diffs = []

    # Track ranking accuracy: count of correct pairs and total pairs
    all_correct_pairs = []
    all_total_pairs = []

    if use_partial_success:
        # RoboArena: Sample pairs of trajectories, rank based on partial_success vs predicted rewards
        for task, trajectories in task_groups.items():
            if len(trajectories) < 2:
                continue

            # Extract ground truth and predicted rewards using vectorized operations
            task_idx = task_indices[task]
            partial_successes = all_partial_successes[task_idx]
            predicted_rewards = all_final_rewards[task_idx]

            # Generate all pairs of trajectories
            n = len(partial_successes)
            if n < 2:
                continue

            # Create pairs: for each pair (i, j) where i < j
            gold_ranks = []
            pred_ranks = []
            correct_pairs = 0
            total_pairs = 0

            for i in range(n):
                for j in range(i + 1, n):
                    # Skip if partial_success values are the same
                    if partial_successes[i] == partial_successes[j]:
                        continue

                    total_pairs += 1

                    # Gold rank: 1 if i has higher partial_success, 0 otherwise
                    gold_rank = 1.0 if partial_successes[i] > partial_successes[j] else 0.0
                    gold_ranks.append(gold_rank)

                    # Predicted rank: 1 if i has higher predicted reward, 0 otherwise
                    if predicted_rewards[i] > predicted_rewards[j]:
                        pred_rank = 1.0
                    elif predicted_rewards[i] < predicted_rewards[j]:
                        pred_rank = 0.0
                    else:
                        # For ties, use 0.5 (middle ground) - count as incorrect
                        pred_rank = 0.5
                    pred_ranks.append(pred_rank)

                    # Check if ranking is correct (gold_rank matches pred_rank)
                    if gold_rank == pred_rank:
                        correct_pairs += 1

            if len(gold_ranks) == 0:
                continue

            # Store ranking accuracy for this task
            if total_pairs > 0:
                task_ranking_acc = correct_pairs / total_pairs
                all_correct_pairs.append(correct_pairs)
                all_total_pairs.append(total_pairs)

            # Compute Spearman correlation between gold and predicted rankings
            spearman_corr = compute_spearman(gold_ranks, pred_ranks)

            # Compute old metric: bin partial_success into 3 bins, then average predicted rewards per bin
            # Create 3 bins based on partial_success values
            min_ps = float(np.min(partial_successes))
            max_ps = float(np.max(partial_successes))

            # Create 3 equal-width bins
            if min_ps == max_ps:
                # All values are the same, put everything in bin 0
                bin_assignments = np.zeros(len(partial_successes), dtype=int)
            else:
                bin_edges = np.linspace(min_ps, max_ps, 4)  # 4 edges = 3 bins
                # Use np.digitize: returns 0 for values < edges[0], 1 for values in [edges[0], edges[1]), etc.
                # We want: bin 0 for [min, bin_edges[1]), bin 1 for [bin_edges[1], bin_edges[2]), bin 2 for [bin_edges[2], max]
                bin_assignments = np.digitize(partial_successes, bin_edges[1:], right=False)
                # But digitize returns 0 for values < first edge, so we need to adjust
                # Actually, digitize with right=False:
                # - values < edges[0] -> 0
                # - edges[0] <= values < edges[1] -> 1
                # - edges[1] <= values < edges[2] -> 2
                # - values >= edges[2] -> 3
                # But we only pass edges[1:] so:
                # - values < edges[1] -> 0
                # - edges[1] <= values < edges[2] -> 1
                # - values >= edges[2] -> 2
                # This gives us bins 0, 1, 2, but we need to handle edge case where value equals max
                bin_assignments = np.clip(bin_assignments, 0, 2)

            avg_rewards_per_bin = {}
            bin_ranks = []
            avg_reward_values = []

            for bin_idx in range(3):
                mask = bin_assignments == bin_idx
                if np.any(mask):
                    bin_rewards = predicted_rewards[mask]
                    avg_reward = float(np.mean(bin_rewards))
                    avg_rewards_per_bin[bin_idx] = avg_reward
                    bin_ranks.append(bin_idx)
                    avg_reward_values.append(avg_reward)

            spearman_rewind = None
            if len(bin_ranks) >= 2:
                spearman_rewind = compute_spearman(bin_ranks, avg_reward_values)
                if not np.isnan(spearman_rewind):
                    all_spearman_rewind.append(spearman_rewind)
                else:
                    spearman_rewind = None

            if not np.isnan(spearman_corr):
                all_spearman.append(spearman_corr)
                task_details[task] = {
                    "spearman": float(spearman_corr),
                    "spearman_rewind": float(spearman_rewind) if spearman_rewind is not None else None,
                }
    else:
        # Non-RoboArena: Use quality_label and quality_order for ranking
        quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}
        all_labels = ["failure", "suboptimal", "successful"]

        for task, trajectories in task_groups.items():
            # Collect rewards per quality using vectorized operations
            task_idx = task_indices[task]
            task_rewards = all_final_rewards[task_idx]
            task_quality_labels = [all_quality_labels[i] for i in task_idx]

            quality_to_rewards = {q: [] for q in all_labels}
            for quality, reward in zip(task_quality_labels, task_rewards):
                quality_to_rewards[quality].append(float(reward))

            present_labels = [q for q in all_labels if quality_to_rewards[q]]

            if len(present_labels) < 2:
                continue

            # Use triplets if all three present, else pairs
            k = len(present_labels)

            spearman = []

            for labels_combo in combinations(present_labels, k):
                gold_ranks = [quality_order[q] for q in labels_combo]

                # All ways to pick one reward per selected quality
                for rewards_tuple in product(*(quality_to_rewards[q] for q in labels_combo)):
                    spearman_corr = compute_spearman(gold_ranks, list(rewards_tuple))
                    if not np.isnan(spearman_corr):
                        spearman.append(spearman_corr)

            avg_spearman_corr = float(np.mean(spearman)) if spearman else 0.0

            # Compute old metric: average rewards per quality group using vectorized operations
            avg_rewards_per_quality = {}
            quality_ranks = []
            avg_reward_values = []
            for q in present_labels:
                rewards = np.array(quality_to_rewards[q])
                if len(rewards) > 0:
                    avg_reward = float(np.mean(rewards))
                    avg_rewards_per_quality[q] = avg_reward
                    quality_ranks.append(quality_order[q])
                    avg_reward_values.append(avg_reward)

            # Compute ranking accuracy: check pairs where quality ordering matches predicted reward ordering
            correct_pairs = 0
            total_pairs = 0

            # Check all pairs of quality labels
            for i, quality1 in enumerate(present_labels):
                for quality2 in present_labels[i + 1 :]:
                    # Get average rewards for each quality
                    avg_reward1 = avg_rewards_per_quality[quality1]
                    avg_reward2 = avg_rewards_per_quality[quality2]

                    # Expected ordering: higher quality should have higher reward
                    expected_order = quality_order[quality1] > quality_order[quality2]
                    actual_order = avg_reward1 > avg_reward2

                    total_pairs += 1
                    if expected_order == actual_order:
                        correct_pairs += 1

            # Store ranking accuracy for this task
            if total_pairs > 0:
                task_ranking_acc = correct_pairs / total_pairs
                all_correct_pairs.append(correct_pairs)
                all_total_pairs.append(total_pairs)

            spearman_rewind = None
            if len(quality_ranks) >= 2:
                spearman_rewind = compute_spearman(quality_ranks, avg_reward_values)
                if not np.isnan(spearman_rewind):
                    all_spearman_rewind.append(spearman_rewind)
                else:
                    spearman_rewind = None

            # Compute average differences between quality labels
            succ_subopt_diff = None
            subopt_fail_diff = None
            succ_fail_diff = None

            if "successful" in avg_rewards_per_quality and "suboptimal" in avg_rewards_per_quality:
                succ_subopt_diff = avg_rewards_per_quality["successful"] - avg_rewards_per_quality["suboptimal"]
                all_succ_subopt_diffs.append(succ_subopt_diff)

            if "suboptimal" in avg_rewards_per_quality and "failure" in avg_rewards_per_quality:
                subopt_fail_diff = avg_rewards_per_quality["suboptimal"] - avg_rewards_per_quality["failure"]
                all_subopt_fail_diffs.append(subopt_fail_diff)

            if "successful" in avg_rewards_per_quality and "failure" in avg_rewards_per_quality:
                succ_fail_diff = avg_rewards_per_quality["successful"] - avg_rewards_per_quality["failure"]
                all_succ_fail_diffs.append(succ_fail_diff)

            task_details[task] = {
                "spearman": avg_spearman_corr,
                "spearman_rewind": spearman_rewind,
                "succ_subopt_diff": succ_subopt_diff,
                "subopt_fail_diff": subopt_fail_diff,
                "succ_fail_diff": succ_fail_diff,
            }
            all_spearman.append(avg_spearman_corr)

    if len(all_spearman) == 0:
        return {"error": "No valid correlations computed"}, {}, {}

    # Compute overall ranking accuracy
    ranking_acc = None
    total_pairs = 0
    if all_total_pairs:
        total_correct = sum(all_correct_pairs)
        total_pairs = sum(all_total_pairs)
        ranking_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    # Average metrics across all tasks
    policy_ranking_metrics = {
        "spearman": np.mean(all_spearman).item(),
        "spearman_rewind": np.mean(all_spearman_rewind).item() if all_spearman_rewind else None,
        "avg_succ_subopt_diff": np.mean(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "min_succ_subopt_diff": np.min(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "max_succ_subopt_diff": np.max(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "avg_subopt_fail_diff": np.mean(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "min_subopt_fail_diff": np.min(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "max_subopt_fail_diff": np.max(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "avg_succ_fail_diff": np.mean(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "min_succ_fail_diff": np.min(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "max_succ_fail_diff": np.max(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "ranking_acc": ranking_acc,
        "ranking_total_pairs": total_pairs,
    }

    return policy_ranking_metrics, task_groups, task_details


def run_similarity_score_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run similarity_score evaluation analysis.

    Groups results by task and computes:
    - Average similarity score for human->same_task (robot)
    - Average similarity score for human->diff_task (negatives)
    - Margin between same_task and diff_task scores
    - Averages results across N negatives per pairing

    Args:
        results: List of evaluation results, each containing:
            - task: Task name
            - sim_score_ref_sim: Similarity score for ref->sim (human->robot, same task)
            - sim_score_ref_diff: Similarity score for ref->diff (human->negative, different task)
            - metadata: Contains task, negative_task, human_id, robot_id, negative_id
        progress_pred_type: Not used for similarity_score, but kept for consistency

    Returns:
        Tuple of (metrics_dict, task_groups, task_details)
    """
    all_sim_scores = []
    all_diff_scores = []
    all_tasks = []
    all_pair_keys = []
    valid_indices = []

    for idx, r in enumerate(results):
        task = r.get("task")
        if task is None:
            continue
        metadata = r.get("metadata", {})
        human_id = metadata.get("human_id")
        robot_id = metadata.get("robot_id")
        if human_id is None or robot_id is None:
            continue

        sim_score_ref_sim = r.get("sim_score_ref_sim")
        sim_score_ref_diff = r.get("sim_score_ref_diff")

        # Convert scores to float
        if sim_score_ref_sim is not None:
            if isinstance(sim_score_ref_sim, np.ndarray):
                sim_score_ref_sim = float(sim_score_ref_sim.item())
            else:
                sim_score_ref_sim = float(sim_score_ref_sim)
        else:
            continue

        if sim_score_ref_diff is not None:
            if isinstance(sim_score_ref_diff, np.ndarray):
                sim_score_ref_diff = float(sim_score_ref_diff.item())
            else:
                sim_score_ref_diff = float(sim_score_ref_diff)
        else:
            continue

        all_sim_scores.append(sim_score_ref_sim)
        all_diff_scores.append(sim_score_ref_diff)
        all_tasks.append(task)
        all_pair_keys.append((human_id, robot_id))
        valid_indices.append(idx)

    if not all_sim_scores:
        return {"error": "No valid similarity score data found"}, {}, {}

    # Convert to numpy arrays for vectorized operations
    all_sim_scores = np.array(all_sim_scores)
    all_diff_scores = np.array(all_diff_scores)

    # Group results by task and pair
    task_groups = defaultdict(list)
    task_pair_indices = defaultdict(lambda: defaultdict(list))

    for i, (idx, task, pair_key) in enumerate(zip(valid_indices, all_tasks, all_pair_keys)):
        task_groups[task].append(results[idx])
        task_pair_indices[task][pair_key].append(i)

    if not task_groups:
        return {"error": "No valid similarity score data found"}, {}, {}

    task_details = {}
    all_margins = []
    all_same_task_scores = []
    all_diff_task_scores = []

    for task, task_results in task_groups.items():
        # Compute metrics per pair, then average across pairs using vectorized operations
        pair_margins = []
        pair_same_task_scores = []
        pair_diff_task_scores = []

        for pair_key, pair_indices in task_pair_indices[task].items():
            pair_sim_scores = all_sim_scores[pair_indices]
            pair_diff_scores = all_diff_scores[pair_indices]

            # Average across negatives for this pair (vectorized)
            avg_same_task = np.mean(pair_sim_scores)
            avg_diff_task = np.mean(pair_diff_scores)
            margin = avg_same_task - avg_diff_task

            pair_margins.append(margin)
            pair_same_task_scores.append(avg_same_task)
            pair_diff_task_scores.append(avg_diff_task)

        if pair_margins:
            # Convert to numpy arrays for vectorized mean computation
            pair_margins = np.array(pair_margins)
            pair_same_task_scores = np.array(pair_same_task_scores)
            pair_diff_task_scores = np.array(pair_diff_task_scores)

            task_margin = float(np.mean(pair_margins))
            task_same_task_score = float(np.mean(pair_same_task_scores))
            task_diff_task_score = float(np.mean(pair_diff_task_scores))

            task_details[task] = {
                "avg_margin": task_margin,
                "avg_same_task_score": task_same_task_score,
                "avg_diff_task_score": task_diff_task_score,
                "num_pairs": len(pair_margins),
            }

            all_margins.append(task_margin)
            all_same_task_scores.append(task_same_task_score)
            all_diff_task_scores.append(task_diff_task_score)

    if not all_margins:
        return {"error": "No valid margins computed"}, {}, {}

    # Aggregate metrics across all tasks
    metrics = {
        "avg_margin": np.mean(all_margins).item(),
        "avg_same_task_score": np.mean(all_same_task_scores).item(),
        "avg_diff_task_score": np.mean(all_diff_task_scores).item(),
    }

    return metrics, task_groups, task_details
