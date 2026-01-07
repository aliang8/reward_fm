#!/usr/bin/env python3
"""
Script to compile evaluation results from JSON files.
"""

import json
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from rfm.data.datasets.helpers import load_frames_from_npz
from rfm.evals.eval_metrics_utils import compute_pearson, compute_spearman
from rfm.evals.eval_viz_utils import create_combined_progress_success_plot
from rfm.models.utils import convert_bins_to_continuous, convert_discrete_target_to_continuous 


def compute_eval_metrics(
    eval_type: str,
    results: List[Dict[str, Any]],
    progress_pred_type: str,
    is_discrete_mode: bool = False,
    num_bins: int = 10,
    data_source: Optional[str] = None,
):
    if eval_type == "quality_preference" or eval_type == "quality_preference_roboarena":
        return run_quality_preference_eval(results, data_source=data_source)
    elif eval_type == "reward_alignment":
        return run_reward_alignment_eval_per_trajectory(
            results, progress_pred_type, is_discrete_mode, num_bins, data_source
        )
    elif eval_type == "confusion_matrix":
        return run_confusion_matrix_eval(results, progress_pred_type, is_discrete_mode, num_bins)
    elif eval_type == "policy_ranking":
        return run_policy_ranking_eval(results, progress_pred_type, is_discrete_mode, num_bins, data_source)
    elif eval_type == "similarity_score":
        return run_similarity_score_eval(results)


def run_quality_preference_eval(results: List[Dict[str, Any]], data_source: Optional[str] = None) -> Dict[str, Any]:
    """Run quality_preference evaluation analysis.

    Groups results by task and quality labels (or partial_success for RoboArena),
    computes preference accuracy per group and aggregate.
    Returns metrics, task_groups, and task_details similar to policy_ranking.
    """
    use_partial_success = data_source and "roboarena" in str(data_source).lower()

    # First, gather all predictions and labels, convert to arrays
    # Note: preference_pred is already binary (0/1) from the trainer
    all_preds = []
    all_labels = []
    all_tasks = []
    all_quality_combos = []
    valid_indices = []

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
    # preference_pred is already binary (0/1), so no sigmoid conversion needed
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = all_preds.astype(float)

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
    results: List[Dict[str, Any]],
    progress_pred_type: str,
    is_discrete_mode: bool,
    num_bins: int,
    data_source: Optional[str],
    last_frame_only: bool = False,
) -> Tuple[Dict[str, Any], List, List, List]:
    """Run reward_alignment evaluation analysis and create plots for each trajectory.

    For failure datasets, we visualize predictions but skip metric computation.

    Returns:
        Tuple of (metrics, plots, video_frames_list, trajectory_progress_data)
        where trajectory_progress_data is a list of progress_pred values
        for each trajectory (one per video in video_frames_list)
    """
    # Check if this is RoboArena (uses partial_success instead of quality_label)
    use_partial_success = data_source and "roboarena" in str(data_source).lower()

    # Determine success availability once at the beginning
    have_success = False
    have_success_labels = False
    have_success_probs = False
    if results and len(results) > 0:
        first_result = results[0]
        have_success = first_result.get("success_pred", None) is not None
        have_success_labels = first_result.get("success_labels", None) is not None
        have_success_probs = first_result.get("success_probs", None) is not None

    unique_trajectory_ids = set()
    loss_per_trajectory = np.zeros(1)
    loss_trajectories = []
    pearson_trajectories = []
    plots = []
    video_frames_list = []
    trajectory_progress_data = []

    # Collect all success_probs and success_labels for AUPRC computation
    all_success_probs = []
    all_success_labels = []

    # Collect absolute deltas between final reward and partial_success for RoboArena
    roboarena_deltas = []

    # Collect success_acc for binary success accuracy
    success_acc_list = []

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
        # Sort by frame_step if available (for frame_steps mode)
        # This orders subsequences from shortest to longest (e.g., [0], [0,1], [0,1,2], ...)
        # Only sort if there are multiple results (indicating frame_steps mode)
        if len(results_for_trajectory) > 1:
            results_for_trajectory.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))

        # Get task and quality label from first result
        task = results_for_trajectory[0]["task"]
        quality_label = results_for_trajectory[0]["quality_label"]
        video_path = results_for_trajectory[0]["video_path"]
        partial_success = results_for_trajectory[0].get("partial_success")

        if is_discrete_mode and partial_success is not None:
            if isinstance(partial_success, torch.Tensor):
                # [num_bins] -> [1, 1, num_bins]
                partial_success_tensor = partial_success[None, None] # to make it 3-dim for convert_discrete_target_to_continuous
            else:
                # number -> [1, 1]
                partial_success_tensor = torch.tensor([partial_success], dtype=torch.float32).unsqueeze(0)
            partial_success = convert_discrete_target_to_continuous(partial_success_tensor, num_bins=num_bins).item()

        # Detect if we're in whole trajectory mode (use_frame_steps=False) or frame_steps mode
        is_whole_trajectory_mode = len(results_for_trajectory) == 1

        # First, gather all predictions and targets for this trajectory
        all_preds = []
        all_targets = []
        all_pred_logits = []  # For discrete mode: collect full logits
        all_target_bins = []  # For discrete mode: collect bin indices
        all_success_preds = []
        all_success_labels_list = []
        all_success_probs_list = []

        if is_whole_trajectory_mode:
            # Whole trajectory mode: one result with full progress prediction
            r = results_for_trajectory[0]
            pred = r.get("progress_pred")
            tgt = r.get("target_progress")

            if pred is not None:
                pred_array = np.array(pred)
                if is_discrete_mode:
                    # Discrete mode: pred is logits [seq_len, num_bins]
                    # Convert to continuous values using weighted sum of bin centers
                    if last_frame_only:
                        # Use last frame's logits
                        all_pred_logits.append(pred_array[-1])
                        if tgt is not None and len(tgt) > 0:
                            all_target_bins.append(int(tgt[-1]))
                        continuous_pred = convert_bins_to_continuous(
                            torch.tensor(pred_array[-1], dtype=torch.float32)
                        ).item()
                        all_preds.append(float(continuous_pred))
                    else:
                        # Use all predictions: convert logits to continuous values
                        if pred_array.ndim > 1:
                            # pred_array is [seq_len, num_bins], convert to continuous
                            continuous_preds = convert_bins_to_continuous(
                                torch.tensor(pred_array, dtype=torch.float32)
                            ).numpy()
                            all_preds = continuous_preds.tolist()
                            # Store logits as list of lists (one per timestep) - same format as frame_steps mode
                            all_pred_logits = pred_array.tolist()
                        else:
                            # Already continuous (shouldn't happen in discrete mode, but handle it)
                            #all_preds = pred_array.tolist()
                            print("Warning: Pred array should not be continuous in discrete mode, breakpointing to debug")
                            breakpoint()
                        if tgt is not None and len(tgt) > 0:
                            tgt_array = np.array(tgt)
                            all_target_bins = [int(t) for t in tgt_array]
                            # Convert target bins to continuous values for comparison
                            all_targets = [(float(t) / (num_bins - 1)) for t in tgt_array]
                else:
                    # Continuous mode: use all predictions directly
                    if last_frame_only:
                        all_preds = [float(pred_array[-1])]
                    else:
                        if pred_array.ndim > 0:
                            all_preds = [float(p) for p in pred_array]
                        else:
                            all_preds = [float(pred_array)]
            else:
                all_preds = [0.0]

            if tgt is not None and len(tgt) > 0 and not is_discrete_mode:
                tgt_array = np.array(tgt)
                if last_frame_only:
                    all_targets = [float(tgt_array[-1])]
                else:
                    if tgt_array.ndim > 0:
                        all_targets = [float(t) for t in tgt_array]
                    else:
                        all_targets = [float(tgt_array)]

            # Optional success predictions (whole arrays)
            succ = r.get("success_pred", None)
            if succ is not None:
                if isinstance(succ, (list, np.ndarray)):
                    succ_array = np.array(succ)
                    if succ_array.ndim > 0:
                        all_success_preds = [float(s) for s in succ_array]
                    else:
                        all_success_preds = [float(succ_array)]
                else:
                    all_success_preds = [float(succ)]

            succ_labels = r.get("success_labels", None)
            if succ_labels is not None:
                if isinstance(succ_labels, (list, np.ndarray)):
                    succ_labels_array = np.array(succ_labels)
                    if succ_labels_array.ndim > 0:
                        all_success_labels_list = [float(s) for s in succ_labels_array]
                    else:
                        all_success_labels_list = [float(succ_labels_array)]
                else:
                    all_success_labels_list = [float(succ_labels)]

            succ_probs = r.get("success_probs", None)
            if succ_probs is not None:
                if isinstance(succ_probs, (list, np.ndarray)):
                    succ_probs_array = np.array(succ_probs)
                    if succ_probs_array.ndim > 0:
                        all_success_probs_list = [float(s) for s in succ_probs_array]
                    else:
                        all_success_probs_list = [float(succ_probs_array)]
                else:
                    all_success_probs_list = [float(succ_probs)]

        else:
            # Frame steps mode: multiple results, one per subsequence
            for timestep, r in enumerate(results_for_trajectory):
                pred = r.get("progress_pred")
                tgt = r.get("target_progress")

                if pred is not None:
                    pred_array = np.array(pred)
                    if is_discrete_mode:
                        # Discrete mode: pred is logits [seq_len, num_bins]
                        # Convert to continuous values using weighted sum of bin centers
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
                                    all_target_bins.append(tgt[-1])
                                else:
                                    all_target_bins.append(tgt[indx])
                        # For visualization: convert logits to continuous value
                        if last_frame_only:
                            continuous_pred = convert_bins_to_continuous(
                                torch.tensor(pred_array[-1], dtype=torch.float32)
                            ).item()
                        else:
                            if timestep >= len(pred_array) - 1:
                                continuous_pred = convert_bins_to_continuous(
                                    torch.tensor(pred_array[-1], dtype=torch.float32)
                                ).item()
                            else:
                                continuous_pred = convert_bins_to_continuous(
                                    torch.tensor(pred_array[timestep], dtype=torch.float32)
                                ).item()
                        all_preds.append(float(continuous_pred))
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
                        # Target is a discrete bin index, convert to continuous value
                        # Convert discrete bin target to a continuous value for logging/metrics
                        if last_frame_only:
                            target_bin = tgt[-1]
                        else:
                            if timestep >= len(tgt) - 1:
                                target_bin = tgt[-1]
                            else:
                                target_bin = tgt[timestep]
                        # convert target_bin to tensor of shape (1, ...)
                        if isinstance(target_bin, torch.Tensor):
                            # [num_bins] -> [1, 1, num_bins]
                            target_bin_tensor = target_bin[None, None] # to make it 3-dim for convert_discrete_target_to_continuous
                        else:
                            # number -> [1, 1]
                            target_bin_tensor = torch.tensor([target_bin]).unsqueeze(0)
                        continuous_target = convert_discrete_target_to_continuous(target_bin_tensor, num_bins=num_bins).item()
                        all_targets.append(continuous_target)
                    else:
                        all_targets.append(float(tgt[-1]))
                else:
                    all_targets.append(0.0)

                # Optional success prediction (binary) from trainer outputs
                succ = r.get("success_pred", None)
                if succ is not None and len(succ) > 0:
                    all_success_preds.append(float(succ[-1]))

                # Optional success labels (ground truth) from trainer outputs
                succ_labels = r.get("success_labels", None)
                if succ_labels is not None and len(succ_labels) > 0:
                    all_success_labels_list.append(float(succ_labels[-1]))

                # Optional success probabilities from trainer outputs
                succ_probs = r.get("success_probs", None)
                if succ_probs is not None and len(succ_probs) > 0:
                    all_success_probs_list.append(float(succ_probs[-1]))

        if len(all_preds) == 0 or len(all_targets) == 0:
            print("No valid predictions or targets found for trajectory: ", trajectory_id)
            continue

        last_preds = np.array(all_preds)
        last_targets = np.array(all_targets)
        last_success = np.array(all_success_preds) if all_success_preds else None
        last_success_labels = np.array(all_success_labels_list) if all_success_labels_list else None
        last_success_probs = np.array(all_success_probs_list) if all_success_probs_list else None

        # Load video frames if video path exists
        frames = None
        if video_path:
            frames = load_frames_from_npz(video_path)
            frames = frames.transpose(0, 3, 1, 2)

            # Resize frames to make them smaller for wandb table display
            resized_frames = []
            for frame in frames:
                frame_resized = cv2.resize(frame.transpose(1, 2, 0), (64, 64))
                resized_frames.append(frame_resized.transpose(2, 0, 1))
            frames = np.array(resized_frames)

        video_frames_list.append(frames)

        if progress_pred_type == "relative":
            last_preds = np.cumsum(last_preds)
            last_targets = np.cumsum(last_targets)

        trajectory_progress_data.append(last_preds.tolist())

        # For RoboArena, compute absolute delta between final reward and partial_success
        if use_partial_success and partial_success is not None:
            final_reward = float(last_preds[-1])
            delta = abs(final_reward - partial_success)
            roboarena_deltas.append(delta)

        # Only compute metrics for successful trajectories
        if quality_label == "successful":
            # Compute loss based on mode
            if is_discrete_mode and all_pred_logits and all_target_bins:
                # Discrete mode: compute cross-entropy loss between logits and target bins
                pred_logits_tensor = torch.tensor(np.array(all_pred_logits), dtype=torch.float32)  # [seq_len, num_bins]
                target_bins_tensor = torch.tensor(all_target_bins)  # [seq_len, num_bins] or [seq_len]
                if len(target_bins_tensor.shape) == 1:
                    target_bins_tensor = target_bins_tensor.long()
                loss_per_timestep = F.cross_entropy(pred_logits_tensor, target_bins_tensor, reduction="none")
                traj_loss = float(loss_per_timestep.mean().item())
            else:
                # Continuous mode: compute MSE loss
                traj_loss = float(np.mean((last_targets - last_preds) ** 2))

            # Compute Pearson correlation
            traj_pearson = compute_pearson(last_targets.tolist(), last_preds.tolist())
            # Handle NaN values
            traj_pearson = float(traj_pearson) if not np.isnan(traj_pearson) else 0.0
        else:
            traj_loss = 0.0
            traj_pearson = 0.0

        # Create a wandb plot for progress predictions and, if available, success predictions
        # Use the shared helper function from eval_viz_utils
        has_success_binary = have_success and last_success is not None and len(last_success) == len(last_preds)

        title = f"Task: {task} - {quality_label}\nLoss: {traj_loss:.3f}, pearson: {traj_pearson:.2f}"
        if partial_success is not None:
            title += f", partial_success: {partial_success:.3f}"

        fig = create_combined_progress_success_plot(
            progress_pred=last_preds,
            num_frames=len(last_preds),
            success_binary=last_success if has_success_binary else None,
            success_probs=last_success_probs if have_success_probs and last_success_probs is not None else None,
            success_labels=last_success_labels if have_success_labels and last_success_labels is not None else None,
            is_discrete_mode=is_discrete_mode,
            title=title,
            loss=traj_loss,
            pearson=traj_pearson,
        )

        plots.append(fig)

        # Compute binary success accuracy
        # For successful trajectories: True if anywhere success_prob > 0.5
        # For non-successful trajectories: True if everywhere success_prob <= 0.5
        is_successful_trajectory = quality_label == "successful"
        if last_success_probs is not None and len(last_success_probs) > 0:
            max_success_prob = float(np.max(last_success_probs))
            if is_successful_trajectory:
                # For successful: True if max > 0.5 (we correctly predict success somewhere)
                traj_success_acc = max_success_prob > 0.5
            else:
                # For non-successful: True if max <= 0.5 (we correctly don't predict success)
                traj_success_acc = max_success_prob <= 0.5
            success_acc_list.append(float(traj_success_acc))
        elif have_success_probs:
            # If we have success_probs available but not for this trajectory, skip
            pass

        # Accumulate metrics only for successful trajectories
        if quality_label == "successful":
            loss_trajectories.append(traj_loss)
            if not np.isnan(traj_pearson):
                pearson_trajectories.append(traj_pearson)

    if len(unique_trajectory_ids) == 0:
        loss_per_trajectory = np.nan
        pearson_per_trajectory = np.nan
    else:
        loss_per_trajectory = np.mean(loss_trajectories).item() if loss_trajectories else np.nan
        pearson_per_trajectory = np.mean(pearson_trajectories).item() if pearson_trajectories else np.nan

    # Compute success_auprc across all collected success predictions and labels
    success_auprc = None
    positive_success_acc = None
    negative_success_acc = None
    if all_success_probs and all_success_labels:
        # Flatten all collected probabilities and labels
        success_probs_flat = np.concatenate(all_success_probs)
        success_labels_flat = np.concatenate(all_success_labels)

        # Compute AUPRC if we have valid data
        if success_probs_flat.size > 0 and len(np.unique(success_labels_flat)) > 1:
            success_auprc = float(average_precision_score(success_labels_flat, success_probs_flat))
        else:
            success_auprc = 0.0

        # Compute positive and negative accuracy
        if success_probs_flat.size > 0:
            # Convert probabilities to binary predictions (threshold at 0.5)
            success_preds_flat = (success_probs_flat > 0.5).astype(float)

            # Compute accuracy for positive samples (where label == 1)
            positive_mask = success_labels_flat == 1
            num_positives = positive_mask.sum()
            if num_positives > 0:
                positive_correct = ((success_preds_flat == success_labels_flat) & positive_mask).sum()
                positive_success_acc = float(positive_correct / num_positives)

            # Compute accuracy for negative samples (where label == 0)
            negative_mask = success_labels_flat == 0
            num_negatives = negative_mask.sum()
            if num_negatives > 0:
                negative_correct = ((success_preds_flat == success_labels_flat) & negative_mask).sum()
                negative_success_acc = float(negative_correct / num_negatives)

    metrics = {
        "loss": loss_per_trajectory,
        "pearson": pearson_per_trajectory,
    }

    if success_auprc is not None:
        metrics["success_auprc"] = success_auprc

    # Add binary success accuracy if available
    if success_acc_list:
        metrics["success_acc"] = float(np.mean(success_acc_list))

    # Add positive and negative success accuracy if available
    if positive_success_acc is not None:
        metrics["positive_success_acc"] = positive_success_acc
    if negative_success_acc is not None:
        metrics["negative_success_acc"] = negative_success_acc

    # Add RoboArena delta metric if available
    if use_partial_success and roboarena_deltas:
        metrics["roboarena_abs_delta"] = float(np.mean(roboarena_deltas))

    return metrics, plots, video_frames_list, trajectory_progress_data


def _extract_trajectory_rewards(
    progress_pred: list | np.ndarray,
    progress_pred_type: str,
    is_discrete_mode: bool,
    aggregation: str = "last",
) -> float:
    """Extract trajectory reward using different aggregation methods.

    Args:
        progress_pred: Progress predictions for a single trajectory.
                       For both continuous and discrete modes: list of floats in [0, 1]
                       (discrete mode uses convert_bins_to_continuous to get continuous values)
        progress_pred_type: "relative" or "absolute"
        is_discrete_mode: Whether predictions came from discrete mode (now converted to continuous)
        aggregation: "last", "sum", or "average"

    Returns:
        Aggregated reward (float)
    """
    # Both discrete and continuous modes now use continuous values
    pred_array = np.array(progress_pred, dtype=np.float32)

    # Apply cumsum if relative (for both discrete and continuous modes)
    if progress_pred_type == "relative":
        pred_array = np.cumsum(pred_array)

    # Apply aggregation (same logic for both modes)
    if aggregation == "last":
        reward = (
            pred_array[-1]
            if pred_array.ndim > 0 and len(pred_array) > 0
            else (pred_array if pred_array.ndim == 0 else 0)
        )
    elif aggregation == "sum":
        reward = np.sum(pred_array)
    elif aggregation == "average":
        reward = np.mean(pred_array)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Always return float (discrete mode now uses continuous values)
    return float(reward)


def _compute_policy_ranking_metrics_roboarena(
    all_rewards: np.ndarray,
    all_partial_successes: np.ndarray,
    all_tasks: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute policy ranking metrics for RoboArena datasets using partial_success.

    Args:
        all_rewards: Array of aggregated rewards
        all_partial_successes: Array of partial_success values (already converted to discrete bins if needed)
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    all_partial_successes = np.array(all_partial_successes)

    # Group by task
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        task_indices[task].append(i)

    if not task_indices:
        return {}, {}

    task_details = {}
    all_correct_pairs = []
    all_total_pairs = []
    all_spearman_rewind = []

    # RoboArena: compute ranking accuracy for pairs based on partial_success vs predicted rewards
    for task, task_idx in task_indices.items():
        if len(task_idx) < 2:
            continue

        partial_successes = all_partial_successes[task_idx]
        predicted_rewards = all_rewards[task_idx]

        n = len(partial_successes)
        if n < 2:
            continue

        correct_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Skip pairs with None values
                if partial_successes[i] is None or partial_successes[j] is None:
                    continue
                if partial_successes[i] == partial_successes[j]:
                    continue

                total_pairs += 1
                # True ranking: partial_success[i] should be ranked higher if it's greater
                true_label = partial_successes[i] > partial_successes[j]
                # Predicted ranking: predicted_rewards[i] should be higher if it's greater
                pred_label = predicted_rewards[i] > predicted_rewards[j]

                if true_label == pred_label:
                    correct_pairs += 1

        # Compute spearman_rewind (binning between 0 and 1)
        # Filter out None values for binning
        valid_mask = np.array([ps is not None for ps in partial_successes])
        if np.any(valid_mask):
            valid_partial_successes = np.array([ps for ps in partial_successes if ps is not None])
            valid_predicted_rewards = predicted_rewards[valid_mask]
            bin_edges = np.linspace(0, 1, 4)  # Creates bins [0, 1/3), [1/3, 2/3), [2/3, 1]
            bin_assignments = np.clip(np.digitize(valid_partial_successes, bin_edges[1:], right=False), 0, 2)
        else:
            bin_assignments = np.array([])
            valid_predicted_rewards = np.array([])

        avg_rewards_per_bin = {}
        bin_ranks = []
        avg_reward_values = []

        for bin_idx in range(3):
            mask = bin_assignments == bin_idx
            if np.any(mask):
                bin_rewards = valid_predicted_rewards[mask]
                avg_reward = float(np.mean(bin_rewards))
                avg_rewards_per_bin[bin_idx] = avg_reward
                bin_ranks.append(bin_idx)
                avg_reward_values.append(avg_reward)

        spearman_rewind = None
        if len(bin_ranks) >= 2:
            spearman_rewind = compute_spearman(bin_ranks, avg_reward_values)
            if not np.isnan(spearman_rewind):
                all_spearman_rewind.append(spearman_rewind)

        if total_pairs > 0:
            all_correct_pairs.append(correct_pairs)
            all_total_pairs.append(total_pairs)
            task_ranking_acc = correct_pairs / total_pairs
            task_details[task] = {
                "ranking_acc": float(task_ranking_acc),
                "spearman_rewind": float(spearman_rewind) if spearman_rewind is not None else None,
            }

    if not all_total_pairs:
        return {}, {}

    ranking_acc = None
    if all_total_pairs:
        total_correct = sum(all_correct_pairs)
        total_pairs = sum(all_total_pairs)
        ranking_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    metrics = {
        "ranking_acc_rba": ranking_acc,
        "spearman_rewind_rba": np.mean(all_spearman_rewind).item() if all_spearman_rewind else None,
    }

    return metrics, task_details


def _compute_policy_ranking_metrics_quality_label(
    all_rewards: np.ndarray,
    all_quality_labels: list[str],
    all_tasks: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute policy ranking metrics for datasets using quality_label.

    Args:
        all_rewards: Array of aggregated rewards
        all_quality_labels: List of quality labels
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    # Group by task
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        task_indices[task].append(i)

    if not task_indices:
        return {}, {}

    task_details = {}
    all_spearman = []
    all_spearman_rewind = []
    all_succ_subopt_diffs = []
    all_subopt_fail_diffs = []
    all_succ_fail_diffs = []
    all_correct_pairs = []
    all_total_pairs = []

    # Non-RoboArena: Use quality_label
    quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}
    all_labels = ["failure", "suboptimal", "successful"]

    for task, task_idx in task_indices.items():
        task_rewards = all_rewards[task_idx]
        task_quality_labels = [all_quality_labels[i] for i in task_idx]

        quality_to_rewards = {q: [] for q in all_labels}
        for quality, reward in zip(task_quality_labels, task_rewards):
            quality_to_rewards[quality].append(float(reward))

        present_labels = [q for q in all_labels if quality_to_rewards[q]]

        if len(present_labels) < 2:
            continue

        k = len(present_labels)
        spearman = []

        for labels_combo in combinations(present_labels, k):
            gold_ranks = [quality_order[q] for q in labels_combo]
            for rewards_tuple in product(*(quality_to_rewards[q] for q in labels_combo)):
                spearman_corr = compute_spearman(gold_ranks, list(rewards_tuple))
                if not np.isnan(spearman_corr):
                    spearman.append(spearman_corr)

        avg_spearman_corr = float(np.mean(spearman)) if spearman else 0.0

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

        correct_pairs = 0
        total_pairs = 0
        for i, quality1 in enumerate(present_labels):
            for quality2 in present_labels[i + 1 :]:
                avg_reward1 = avg_rewards_per_quality[quality1]
                avg_reward2 = avg_rewards_per_quality[quality2]
                expected_order = quality_order[quality1] > quality_order[quality2]
                actual_order = avg_reward1 > avg_reward2
                total_pairs += 1
                if expected_order == actual_order:
                    correct_pairs += 1

        if total_pairs > 0:
            all_correct_pairs.append(correct_pairs)
            all_total_pairs.append(total_pairs)

        spearman_rewind = None
        if len(quality_ranks) >= 2:
            spearman_rewind = compute_spearman(quality_ranks, avg_reward_values)
            if not np.isnan(spearman_rewind):
                all_spearman_rewind.append(spearman_rewind)

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
        return {}, {}

    ranking_acc = None
    if all_total_pairs:
        total_correct = sum(all_correct_pairs)
        total_pairs = sum(all_total_pairs)
        ranking_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    metrics = {
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
    }

    return metrics, task_details


def _compute_policy_ranking_metrics_from_rewards(
    all_rewards: np.ndarray,
    use_partial_success: bool,
    all_partial_successes: Optional[np.ndarray],
    all_quality_labels: Optional[List[str]],
    all_tasks: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute policy ranking metrics from pre-computed trajectory rewards.

    Args:
        all_rewards: Array of aggregated rewards
        use_partial_success: Whether this is RoboArena (uses partial_success)
        all_partial_successes: Array of partial_success values (already converted to discrete bins if needed)
        all_quality_labels: List of quality labels (if not use_partial_success)
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    if use_partial_success and all_partial_successes is not None:
        return _compute_policy_ranking_metrics_roboarena(all_rewards, all_partial_successes, all_tasks)
    else:
        return _compute_policy_ranking_metrics_quality_label(all_rewards, all_quality_labels, all_tasks)


def run_confusion_matrix_eval(
    results: List[Dict[str, Any]], progress_pred_type: str, is_discrete_mode: bool, num_bins: int
) -> Dict[str, Any]:
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


def run_policy_ranking_eval(
    results: List[Dict[str, Any]],
    progress_pred_type: str,
    is_discrete_mode: bool,
    num_bins: int,
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Run policy_ranking evaluation analysis.

    Groups results by trajectory_id (like reward_alignment) and computes policy ranking metrics
    using "last", "average", and "sum" aggregation methods.

    For non-RoboArena: Uses quality_label and quality_order for ranking.
    For RoboArena: Uses partial_success for ranking (no quality_order computation).
    """
    # Check if this is RoboArena (uses partial_success instead of quality_label)
    use_partial_success = data_source and "roboarena" in str(data_source).lower()

    # Group results by trajectory_id
    unique_trajectory_ids = set()
    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)

    if not unique_trajectory_ids:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Collect progress predictions per trajectory
    trajectory_progress_preds = {}  # trajectory_id -> list of progress_pred arrays
    trajectory_metadata = {}  # trajectory_id -> {task, quality_label/partial_success, video_path}

    for trajectory_id in unique_trajectory_ids:
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]

        # Sort by frame_step if available (for frame_steps mode)
        # This orders subsequences from shortest to longest (e.g., [0], [0,1], [0,1,2], ...)
        # Only sort if there are multiple results (indicating frame_steps mode)
        if len(results_for_trajectory) > 1:
            results_for_trajectory.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))

        # Collect all progress predictions for this trajectory
        traj_progress_preds = [
            r.get("progress_pred") for r in results_for_trajectory if r.get("progress_pred") is not None
        ]
        trajectory_progress_preds[trajectory_id] = traj_progress_preds

        metadata = {
            "task": results_for_trajectory[0].get("task"),
            "video_path": results_for_trajectory[0].get("video_path"),
            "partial_success": results_for_trajectory[0].get("partial_success"),
            "quality_label": results_for_trajectory[0].get("quality_label"),
        }
        trajectory_metadata[trajectory_id] = metadata

    if not trajectory_progress_preds:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Compute rewards for each trajectory using different aggregation methods
    all_rewards_last = []
    all_rewards_avg = []
    all_rewards_sum = []
    all_tasks = []
    all_quality_labels = []
    all_partial_successes = []
    all_video_paths = []
    all_ids = []

    for trajectory_id, progress_preds_list in trajectory_progress_preds.items():
        metadata = trajectory_metadata[trajectory_id]

        # Process progress predictions: convert logits to continuous values if needed
        processed_progress_preds = []
        for progress_pred in progress_preds_list:
            pred_array = np.array(progress_pred)

            if is_discrete_mode:
                # Discrete mode: pred_array might be logits [seq_len, num_bins]
                # Convert to continuous values using weighted sum of bin centers
                if pred_array.ndim > 1:
                    # It's logits [seq_len, num_bins], convert to continuous values
                    continuous_preds = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).numpy()
                    processed_progress_preds.append(continuous_preds.tolist())
                elif pred_array.ndim == 1:
                    # Single frame logits [num_bins], convert to continuous
                    continuous_pred = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).item()
                    processed_progress_preds.append([float(continuous_pred)])
                else:
                    # Scalar (shouldn't happen, but handle it)
                    processed_progress_preds.append([float(pred_array)])
            else:
                # Continuous mode: pred_array is scalar values
                if pred_array.ndim > 0:
                    processed_progress_preds.append(pred_array.tolist())
                else:
                    processed_progress_preds.append([float(pred_array)])

        if not processed_progress_preds:
            continue

        # Take the last prediction from each subsequence (e.g., if max_frames=4, take the 4th prediction)
        # Then use _extract_trajectory_rewards to compute rewards with different aggregation methods
        last_predictions = []
        for pred_list in processed_progress_preds:
            last_predictions.append(pred_list[-1])

        if not last_predictions:
            continue

        # Use _extract_trajectory_rewards with the list of last predictions from each subsequence
        reward_last = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="last",
        )
        reward_avg = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="average",
        )
        reward_sum = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="sum",
        )

        # Skip trajectories with None partial_success for RoboArena
        if use_partial_success:
            if metadata["partial_success"] is None:
                continue
            if is_discrete_mode:
                if isinstance(metadata["partial_success"], torch.Tensor):
                    # [num_bins] -> [1, 1, num_bins]
                    partial_success_tensor = metadata["partial_success"][None, None] # to make it 3-dim for convert_discrete_target_to_continuous
                else:
                    # number -> [1, 1]
                    partial_success_tensor = torch.tensor([metadata["partial_success"]], dtype=torch.float32).unsqueeze(0)
                metadata["partial_success"] = convert_discrete_target_to_continuous(partial_success_tensor, num_bins=num_bins).item()
            all_partial_successes.append(metadata["partial_success"])
        else:
            all_quality_labels.append(metadata["quality_label"])

        all_rewards_last.append(reward_last)
        all_rewards_avg.append(reward_avg)
        all_rewards_sum.append(reward_sum)

        all_tasks.append(metadata["task"])
        all_video_paths.append(metadata.get("video_path"))
        all_ids.append(trajectory_id)

    all_rewards_last = np.array(all_rewards_last)
    all_rewards_avg = np.array(all_rewards_avg)
    all_rewards_sum = np.array(all_rewards_sum)

    # Group by task for building task_groups (for return value)
    task_groups = {}
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        if task not in task_groups:
            task_groups[task] = []

        task_entry = {
            "final_predicted_reward_last": all_rewards_last[i],
            "final_predicted_reward_avg": all_rewards_avg[i],
            "final_predicted_reward_sum": all_rewards_sum[i],
            "video_path": all_video_paths[i],
        }

        # Add id if available
        if all_ids:
            task_entry["id"] = all_ids[i]

        # Add specific key based on dataset type
        if use_partial_success:
            task_entry["partial_success"] = all_partial_successes[i]
        else:
            task_entry["quality_label"] = all_quality_labels[i]

        task_groups[task].append(task_entry)
        task_indices[task].append(i)

    if not task_groups:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Compute policy ranking metrics for each aggregation method
    all_metrics = {}
    all_task_details = {}

    for agg_method, rewards in [("last", all_rewards_last), ("avg", all_rewards_avg), ("sum", all_rewards_sum)]:
        metrics, task_details = _compute_policy_ranking_metrics_from_rewards(
            rewards,
            use_partial_success,
            np.array(all_partial_successes) if use_partial_success and all_partial_successes else None,
            all_quality_labels if not use_partial_success else None,
            all_tasks,
        )

        if metrics:
            # Prefix metrics with aggregation method
            prefixed_metrics = {f"{k}_{agg_method}" if k != "error" else k: v for k, v in metrics.items()}
            all_metrics.update(prefixed_metrics)

            # Merge task details (keep first one, or combine if needed)
            if not all_task_details:
                all_task_details = task_details
            else:
                # Merge task details by adding aggregation suffix to keys
                for task, details in task_details.items():
                    if task not in all_task_details:
                        all_task_details[task] = {}
                    for k, v in details.items():
                        all_task_details[task][f"{k}_{agg_method}"] = v

    if not all_metrics:
        return {"error": "No valid correlations computed"}, {}, {}

    return all_metrics, task_groups, all_task_details


def run_similarity_score_eval(results: list[dict[str, Any]]) -> dict[str, Any]:
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
