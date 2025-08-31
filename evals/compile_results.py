#!/usr/bin/env python3
"""
Script to compile evaluation results from JSON files.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, List, Any
import matplotlib.patches as patches
from PIL import Image
import io
import base64
from pathlib import Path
from rfm.utils.video_utils import extract_frames_from_video


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compute_pearson(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Pearson correlation, robust to constant inputs; returns np.nan if undefined."""
    import numpy as _np
    from scipy.stats import pearsonr as _pearsonr

    a = _np.asarray(y_true, dtype=float)
    b = _np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _np.nan
    # If either vector is constant, pearsonr returns nan; keep that behavior
    try:
        corr, _ = _pearsonr(a, b)
    except Exception:
        corr = _np.nan
    return corr


def compute_spearman(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Spearman correlation, robust to constant inputs; returns np.nan if undefined."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    try:
        corr, _ = spearmanr(a, b)
    except Exception:
        corr = np.nan
    return corr


def compute_preference_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute preference accuracy over a list of result dicts.
    Expects keys 'predicted_preference' and 'preference_label' per sample.
    Returns dict with accuracy, correct, total, and skipped counts.
    """
    correct = 0
    total = 0
    skipped = 0
    for r in results:
        pred = r.get("predicted_preference")
        label = r.get("preference_label")
        if pred is None or label is None:
            skipped += 1
            continue
        if pred == label:
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else None
    return {
        "preference_accuracy": acc,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
    }


def main():
    import yaml
    from rfm.configs.experiment_configs import DataConfig
    from rfm.configs.eval_configs import EvaluationConfig
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compile evaluation results and create visualizations")
    parser.add_argument(
        "--config", type=str, default="rfm/configs/eval_config.yaml", help="Path to evaluation configuration file"
    )
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing multiple results JSONs")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to visualize")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    # Directory mode: process known files with tailored analyses
    if args.results_dir is not None:
        dir_path = Path(args.results_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"results_dir not found or not a directory: {dir_path}")

        print(f"Scanning results directory: {dir_path}")
        available = {p.name: p for p in dir_path.glob("*.json")}
        print(f"Found JSON files: {sorted(list(available.keys()))}")

        # Helper to safely load a file
        def _load_if_exists(name: str) -> List[Dict[str, Any]]:
            if name in available:
                print(f"Loading {name}...")
                return load_results(str(available[name]))
            print(f"Skipping {name}: file not found")
            return []

        # success_failure.json: compute Pearson/Spearman for success trajectory + preference accuracy
        sf_results = _load_if_exists("success_failure.json")
        if sf_results:
            print("Running analyses for success_failure.json:")

            def _extract_series(results: List[Dict[str, Any]]):
                y_true_all = []
                y_pred_all = []
                for r in results:
                    pred = r.get("progress_pred_chosen")
                    meta = r.get("chosen_metadata", {}) or {}
                    tgt = meta.get("target_progress")
                    if pred is not None and tgt is not None:
                        y_pred_all.extend(list(pred))
                        y_true_all.extend(list(tgt))
                return y_true_all, y_pred_all

            y_true_sf, y_pred_sf = _extract_series(sf_results)
            pearson_sf = compute_pearson(y_true_sf, y_pred_sf)
            spearman_sf = compute_spearman(y_true_sf, y_pred_sf)
            pref_acc_sf = compute_preference_accuracy(sf_results)
            print(f"  - Success trajectory Pearson: {pearson_sf if not np.isnan(pearson_sf) else 'nan'}")
            print(f"  - Success trajectory Spearman: {spearman_sf if not np.isnan(spearman_sf) else 'nan'}")
            print(
                f"  - Preference accuracy: {pref_acc_sf['preference_accuracy'] if pref_acc_sf['preference_accuracy'] is not None else 'N/A'}"
                f" (correct={pref_acc_sf['num_correct']}, total={pref_acc_sf['num_total']}, skipped={pref_acc_sf['num_skipped']})"
            )

        # reward_alignment.json: reuse compute_metrics and summaries
        ra_results = _load_if_exists("reward_alignment.json")
        if ra_results:
            print("Running analyses for reward_alignment.json:")

            last_preds = []
            last_targets = []
            for r in ra_results:
                pred = r.get("progress_pred_A")
                meta = r.get("metadata", {}) or {}
                tgt = meta.get("target_progress")
                if pred and len(pred) > 0 and tgt and len(tgt) > 0:
                    last_preds.append(float(pred[-1]))
                    last_targets.append(float(tgt[-1]))

            mse = np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
            pearson_last = compute_pearson(last_targets, last_preds)
            spearman_last = compute_spearman(last_targets, last_preds)
            print("  - MSE:", mse)
            print("  - Pearson:", pearson_last if not np.isnan(pearson_last) else "nan")
            print("  - Spearman:", spearman_last if not np.isnan(spearman_last) else "nan")
        else:
            print("No analyses run for reward_alignment.json")

        # confusion_matrix.json: create confusion matrix analysis
        cm_results = _load_if_exists("confusion_matrix_progress.json")
        if cm_results:
            print("Running analyses for confusion_matrix_progress.json:")

            # Group results by confusion matrix task and trajectory original task
            task_task_groups = {}

            for r in cm_results:
                # Get the confusion matrix task (the task we're testing with) and original trajectory task
                meta = r.get("metadata", {}) or {}
                cm_task = meta.get("confusion_matrix_task")  # Language task given to model
                trajectory_original_task = meta.get("trajectory_original_task")  # Original task of trajectory

                if cm_task is None or trajectory_original_task is None:
                    continue

                # Get the final progress prediction as the reward
                progress_pred = r.get("progress_pred_A", [])
                if not progress_pred or len(progress_pred) == 0:
                    continue

                final_reward = float(progress_pred[-1])

                # Group by (cm_task, trajectory_original_task) pair
                key = (cm_task, trajectory_original_task)
                if key not in task_task_groups:
                    task_task_groups[key] = []
                task_task_groups[key].append(final_reward)

            print(f"  - Found {len(task_task_groups)} task-task pairs")

            # Get unique tasks for both axes
            all_tasks = set()
            for cm_task, traj_task in task_task_groups.keys():
                all_tasks.add(cm_task)
                all_tasks.add(traj_task)

            unique_tasks = sorted(list(all_tasks))
            print(f"  - Found {len(unique_tasks)} unique tasks: {unique_tasks}")

            # Create confusion matrix: rows = cm_task (language given to model), cols = trajectory_original_task
            confusion_matrix = np.zeros((len(unique_tasks), len(unique_tasks)))

            # Fill confusion matrix with average rewards
            for i, cm_task in enumerate(unique_tasks):
                for j, traj_task in enumerate(unique_tasks):
                    key = (cm_task, traj_task)
                    if key in task_task_groups:
                        # Average across all trajectories for this task-task pair
                        avg_reward = np.mean(task_task_groups[key])
                        confusion_matrix[i, j] = avg_reward
                    else:
                        confusion_matrix[i, j] = np.nan

            # Print confusion matrix summary
            print(f"  - Confusion matrix shape: {confusion_matrix.shape}")

            plt.figure(figsize=(12, 10))
            plt.imshow(confusion_matrix, cmap="viridis", aspect="auto")
            plt.colorbar(label="Average Final Progress Reward")
            plt.xlabel("Trajectory Original Task")
            plt.ylabel("Language Task Given to Model")
            plt.title("Confusion Matrix: Language Task vs Trajectory Original Task")
            plt.xticks(range(len(unique_tasks)), unique_tasks, rotation=45, ha="right")
            plt.yticks(range(len(unique_tasks)), unique_tasks)
            plt.tight_layout()

            # Save plot
            plot_path = dir_path / "confusion_matrix_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"  - Saved confusion matrix heatmap to: {plot_path}")
            plt.close()

        else:
            print("No analyses run for confusion_matrix.json")

        print("Directory processing complete.")
        print("Done!")
        return


if __name__ == "__main__":
    main()
