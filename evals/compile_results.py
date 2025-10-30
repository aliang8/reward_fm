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

from rfm.configs.eval_configs import EvaluationConfig
from rfm.configs.experiment_configs import DataConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from rfm.data.datasets.helpers import load_frames_from_npz
from evals.eval_metrics_utils import compute_pearson, compute_preference_accuracy, compute_spearman


def load_results(results_path: str) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def compute_eval_metrics(eval_type: str, results: list[dict[str, Any]], progress_pred_type: str):
    if eval_type == "success_failure_preference" or eval_type == "success_failure":
        return run_success_failure_eval(results, progress_pred_type)
    elif eval_type == "reward_alignment_progress" or eval_type == "reward_alignment":
        return run_reward_alignment_eval_per_trajectory(results, progress_pred_type)
    elif eval_type == "confusion_matrix_progress" or eval_type == "confusion_matrix":
        return run_confusion_matrix_eval(results, progress_pred_type)
    elif eval_type == "wrong_task_preference" or eval_type == "wrong_task":
        return run_success_failure_eval(results, progress_pred_type)
    elif eval_type == "policy_ranking_progress" or eval_type == "policy_ranking":
        return run_policy_ranking_eval_per_ranked_set(results, progress_pred_type)


def run_success_failure_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run success_failure evaluation analysis."""

    def _extract_series(results: list[dict[str, Any]]):
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

    _y_true_sf, _y_pred_sf = _extract_series(results)
    pref_acc_sf = compute_preference_accuracy(results)

    return {
        "preference_accuracy": pref_acc_sf["preference_accuracy"],
        "num_correct": pref_acc_sf["num_correct"],
        "num_total": pref_acc_sf["num_total"],
        "num_skipped": pref_acc_sf["num_skipped"],
        "num_samples": len(results),
    }


def run_reward_alignment_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run reward_alignment evaluation analysis."""
    last_preds = []
    last_targets = []
    for r in results:
        pred = r.get("progress_pred")
        if len(pred) == 0:
            continue
        tgt = r.get("target_progress")
        last_preds.append(float(pred[-1]))
        last_targets.append(float(tgt[-1]))

    if not last_preds or not last_targets:
        return {"error": "No valid predictions or targets found"}

    if progress_pred_type == "relative":
        last_preds = np.cumsum(last_preds)
        last_targets = np.cumsum(last_targets)

    mse = np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
    pearson = compute_pearson(last_targets, last_preds)
    spearman = compute_spearman(last_targets, last_preds)

    return {
        "mse": mse.item(),
        "pearson": pearson.item() if not np.isnan(pearson) else 0.0,
        "spearman": spearman.item() if not np.isnan(spearman) else 0.0,
        # "num_samples": len(last_preds),
    }


def run_reward_alignment_eval_per_trajectory(
    results: list[dict[str, Any]], progress_pred_type: str
) -> tuple[dict[str, Any], list, list]:
    """Run reward_alignment evaluation analysis and create plots for each trajectory."""
    unique_trajectory_ids = set()
    mse_per_trajectory = 0
    pearson_trajectories = []
    spearman_trajectories = []
    plots = []
    video_frames_list = []

    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)

    for trajectory_id in unique_trajectory_ids:
        last_preds = []
        last_targets = []
        progress_preds = []
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]
        results_for_trajectory.sort(key=lambda r: r["metadata"]["subsequence_end"])

        # Get task and quality label from first result
        task = results_for_trajectory[0].get("task", "unknown")
        quality_label = results_for_trajectory[0].get("quality_label", "unknown")

        if quality_label != "successful":
            continue

        # Try to get video_path from results, if not available, we'll return None for frames
        video_path = results_for_trajectory[0].get("video_path", None)

        for r in results_for_trajectory:
            pred = r.get("progress_pred")
            tgt = r.get("target_progress")
            r.get("metadata", {})
            if pred is not None:
                last_preds.append(float(pred[-1]))
            else:
                last_preds.append(0.0)
            if tgt is not None:
                last_targets.append(float(tgt[-1]))
            else:
                last_targets.append(0.0)

        if len(last_preds) == 0 or len(last_targets) == 0:
            print("No valid predictions or targets found for trajectory: ", trajectory_id)
            continue

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

        # Calculate metrics for this trajectory
        traj_mse = np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
        traj_pearson = compute_pearson(last_targets, last_preds)
        traj_spearman = compute_spearman(last_targets, last_preds)

        # Handle NaN values
        traj_pearson = traj_pearson if not np.isnan(traj_pearson) else 0.0
        traj_spearman = traj_spearman if not np.isnan(traj_spearman) else 0.0

        # Create a wandb plot for progress predictions similar to the custom eval
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(last_preds, linewidth=2)
        ax.set_ylabel("Progress")
        ax.set_title(
            f"Progress Pred - {task} - {quality_label}\nMSE: {traj_mse:.2f}, Pearson: {traj_pearson:.2f}, Spearman: {traj_spearman:.2f}"
        )
        ax.set_ylim(0, 1)
        # remove right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # remove y ticks
        ax.set_yticks([])

        plots.append(fig)

        mse_per_trajectory += np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
        pearson = compute_pearson(last_targets, last_preds)
        if not np.isnan(pearson):
            pearson_trajectories.append(pearson)
        spearman = compute_spearman(last_targets, last_preds)
        if not np.isnan(spearman):
            spearman_trajectories.append(spearman)

    mse_per_trajectory = mse_per_trajectory / len(unique_trajectory_ids)
    pearson_per_trajectory = np.mean(pearson_trajectories)
    spearman_per_trajectory = np.mean(spearman_trajectories)

    metrics = {
        "mse": mse_per_trajectory.item(),
        "pearson": pearson_per_trajectory.item(),
        "spearman": spearman_per_trajectory.item(),
        "num_samples": len(unique_trajectory_ids),
    }

    return metrics, plots, video_frames_list


def run_confusion_matrix_eval(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run confusion_matrix evaluation analysis."""
    # Group results by confusion matrix task
    uniq_tasks = set()

    for r in results:
        meta = r["metadata"]
        lang_task = meta["lang_task"]
        video_task = meta["video_task"]
        uniq_tasks.add(lang_task)
        uniq_tasks.add(video_task)

    task_to_idx = {task: idx for idx, task in enumerate(uniq_tasks)}

    num_tasks = len(uniq_tasks)
    confusion_matrix = np.zeros((num_tasks, num_tasks))
    count_matrix = np.zeros((num_tasks, num_tasks))

    for r in results:
        meta = r["metadata"]
        lang_task = meta["lang_task"]
        video_task = meta["video_task"]

        # Get the final progress prediction as the reward
        progress_pred = r["progress_pred"]
        if len(progress_pred) == 0:
            continue
        if progress_pred_type == "relative":
            progress_pred = np.cumsum(progress_pred)
        final_reward = float(progress_pred[-1])

        lang_task_idx = task_to_idx[lang_task]
        video_task_idx = task_to_idx[video_task]
        confusion_matrix[lang_task_idx, video_task_idx] += final_reward
        count_matrix[lang_task_idx, video_task_idx] += 1

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


def run_policy_ranking_eval_per_ranked_set(results: list[dict[str, Any]], progress_pred_type: str) -> dict[str, Any]:
    """Run policy_ranking evaluation analysis per ranked set."""
    task_groups = {}

    for r in results:
        task = r["task"]
        quality_label = r["quality_label"]
        progress_pred = r["progress_pred"]
        if len(progress_pred) == 0:
            continue
        if progress_pred_type == "relative":
            progress_pred = np.cumsum(progress_pred)
        final_reward = float(progress_pred[-1])

        if task not in task_groups:
            task_groups[task] = []

        task_groups[task].append({
            "quality_label": quality_label,
            "final_reward": final_reward,
            "video_path": r["video_path"] if "video_path" in r else None,
        })

    if not task_groups:
        return {"error": "No valid policy ranking data found"}

    task_details = {}

    quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}
    all_labels = ["failure", "suboptimal", "successful"]

    for task, trajectories in task_groups.items():
        # Collect rewards per quality
        quality_to_rewards = {q: [] for q in all_labels}
        for t in trajectories:
            quality_to_rewards[t["quality_label"]].append(t["final_reward"])

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

        avg_spearman_corr = np.mean(spearman)
        task_details[task] = {"spearman": avg_spearman_corr, "num_triplets": len(spearman)}

    # average metrics across all task details
    policy_ranking_metrics = {
        "spearman": np.mean([t["spearman"] for t in task_details.values()]).item(),
        "num_triplets": np.mean([t["num_triplets"] for t in task_details.values()]).item(),
    }

    return policy_ranking_metrics, task_groups, task_details


def main():
    parser = argparse.ArgumentParser(description="Compile evaluation results and create visualizations")
    parser.add_argument(
        "--config", type=str, default="rfm/configs/eval_config.yaml", help="Path to evaluation configuration file"
    )
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing multiple results JSONs")
    parser.add_argument("--eval_logs_dir", type=str, default=None, help="Root directory containing eval_logs structure")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    # Auto-scan eval_logs directory structure
    if args.eval_logs_dir is not None:
        eval_logs_path = Path(args.eval_logs_dir)
        if not eval_logs_path.exists():
            print(f"eval_logs directory not found: {eval_logs_path}")
            return

        print(f"Scanning eval_logs directory: {eval_logs_path}")

        # Find all model directories
        model_dirs = [d for d in eval_logs_path.iterdir() if d.is_dir()]
        print(f"Found model directories: {[d.name for d in model_dirs]}")

        all_results = {}

        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\nProcessing model: {model_name}")

            # Find all dataset directories
            dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            print(f"  Found dataset directories: {[d.name for d in dataset_dirs]}")

            model_results = {}

            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                print(f"  Processing dataset: {dataset_name}")

                # Find all evaluation JSON files
                eval_files = list(dataset_dir.glob("*.json"))
                print(f"    Found evaluation files: {[f.stem for f in eval_files]}")

                dataset_results = {}

                for eval_file in eval_files:
                    eval_type = eval_file.stem  # filename without extension

                    results = load_results(str(eval_file))
                    metrics = compute_eval_metrics(eval_type, results, cfg.data.progress_pred_type)

                    if eval_type == "confusion_matrix_progress" or eval_type == "confusion_matrix":
                        # save the figure
                        fig, confusion_matrix = metrics
                        fig.savefig(eval_file.with_suffix(".png"))
                        # save the confusion matrix
                        np.save(eval_file.with_suffix(".npy"), confusion_matrix)
                    else:
                        dataset_results[eval_type] = metrics

                model_results[dataset_name] = dataset_results

            all_results[model_name] = model_results

        # Save comprehensive results
        results_summary_path = eval_logs_path / "evaluation_summary.json"

        with open(results_summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved comprehensive evaluation summary to: {results_summary_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        for model_name, model_results in all_results.items():
            print(f"\nModel: {model_name}")
            print("-" * 40)

            for dataset_name, dataset_results in model_results.items():
                print(f"  Dataset: {dataset_name}")

                for eval_type, metrics in dataset_results.items():
                    if "error" in metrics:
                        print(f"    {eval_type}: ERROR - {metrics['error']}")
                    else:
                        print(f"    {eval_type}:")
                        for metric_name, value in metrics.items():
                            if isinstance(value, float):
                                print(f"      {metric_name}: {value:.4f}")
                            else:
                                print(f"      {metric_name}: {value}")

        print("\nDone!")
        return

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"results_dir not found: {results_dir}")
            return

        print(f"Loading results from: {results_dir}")
        eval_files = list(results_dir.glob("*.json"))
        print(f"Found evaluation files: {[f.stem for f in eval_files]}")

        for eval_file in eval_files:
            eval_type = eval_file.stem  # filename without extension
            print(f"    Analyzing {eval_type}...")
            results = load_results(str(eval_file))
            metrics = compute_eval_metrics(eval_type, results, cfg.data.progress_pred_type)
            print(f"      ✓ Completed {eval_type} analysis")
            print(f"      ✓ Metrics: {metrics}")
        print("Done!")
        return


if __name__ == "__main__":
    main()
