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
from itertools import combinations, product
from PIL import Image
import io
import base64
from pathlib import Path
from evals.eval_metrics_utils import compute_pearson, compute_spearman, compute_preference_accuracy


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def analyze_evaluation_type(eval_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results based on evaluation type."""
    if eval_type == "success_failure_preference":
        return run_success_failure_eval(results)
    elif eval_type == "reward_alignment_progress":
        return run_reward_alignment_eval(results)
    elif eval_type == "confusion_matrix":
        return run_confusion_matrix_eval(results)
    elif eval_type == "wrong_task_preference":
        return run_success_failure_eval(results)
    elif eval_type == "policy_ranking_progress":
        return run_policy_ranking_eval(results)


def run_success_failure_eval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run success_failure evaluation analysis."""

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

    y_true_sf, y_pred_sf = _extract_series(results)
    pref_acc_sf = compute_preference_accuracy(results)

    return {
        "preference_accuracy": pref_acc_sf["preference_accuracy"],
        "num_correct": pref_acc_sf["num_correct"],
        "num_total": pref_acc_sf["num_total"],
        "num_skipped": pref_acc_sf["num_skipped"],
        "num_samples": len(results),
    }


def run_reward_alignment_eval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run reward_alignment evaluation analysis."""
    last_preds = []
    last_targets = []
    for r in results:
        pred = r.get("progress_pred_A")
        tgt = r.get("target_progress")
        meta = r.get("metadata", {})
        if pred and len(pred) > 0 and tgt and len(tgt) > 0:
            last_preds.append(float(pred[-1]))
            last_targets.append(float(tgt[-1]))

    if not last_preds or not last_targets:
        return {"error": "No valid predictions or targets found"}

    mse = np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
    pearson_last = compute_pearson(last_targets, last_preds)
    spearman_last = compute_spearman(last_targets, last_preds)

    return {
        "mse": mse,
        "pearson_correlation": pearson_last if not np.isnan(pearson_last) else None,
        "spearman_correlation": spearman_last if not np.isnan(spearman_last) else None,
        "num_samples": len(last_preds),
    }


def run_reward_alignment_eval_per_trajectory(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run reward_alignment evaluation analysis."""
    unique_trajectory_ids = set()
    mse_per_trajectory = 0
    pearson_trajectories = []
    spearman_trajectories = []
    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)
    for trajectory_id in unique_trajectory_ids:
        last_preds = []
        last_targets = []
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]
        results_for_trajectory.sort(key=lambda r: r["metadata"]["subsequence_end"])
        for r in results_for_trajectory:
            pred = r.get("progress_pred_A")
            tgt = r.get("target_progress")
            meta = r.get("metadata", {})
            if pred and len(pred) > 0 and tgt and len(tgt) > 0:
                last_preds.append(float(pred[-1]))
                last_targets.append(float(tgt[-1]))
        if not last_preds or not last_targets:
            print("No valid predictions or targets found for trajectory: ", trajectory_id)
            continue

        mse_per_trajectory += np.mean((np.array(last_targets) - np.array(last_preds)) ** 2)
        pearson = compute_pearson(last_targets, last_preds)
        if not np.isnan(pearson):
            pearson_trajectories.append(pearson)
        spearman = compute_spearman(last_targets, last_preds)
        if not np.isnan(spearman):
            spearman_trajectories.append(spearman)

    # import pdb; pdb.set_trace()
    mse_per_trajectory = mse_per_trajectory / len(unique_trajectory_ids)
    pearson_per_trajectory = np.mean(pearson_trajectories)
    spearman_per_trajectory = np.mean(spearman_trajectories)

    return {
        "mse": mse_per_trajectory,
        "pearson_correlation": pearson_per_trajectory,
        "spearman_correlation": spearman_per_trajectory,
        "num_samples": len(unique_trajectory_ids),
    }


def run_confusion_matrix_eval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run confusion_matrix evaluation analysis."""
    # Group results by confusion matrix task
    task_groups = {}
    trajectory_rewards = {}

    for r in results:
        # Get the confusion matrix task (the task we're testing with)
        chosen_meta = r.get("chosen_meta", {}) or {}
        cm_task = chosen_meta.get("confusion_matrix_task")
        trajectory_original_task = chosen_meta.get("trajectory_original_task")
        trajectory_id = chosen_meta.get("id", "unknown")

        if cm_task is None:
            continue

        # Get the final progress prediction as the reward
        progress_pred = r.get("progress_pred_chosen", [])
        if not progress_pred or len(progress_pred) == 0:
            continue

        final_reward = float(progress_pred[-1])

        # Group by confusion matrix task
        if cm_task not in task_groups:
            task_groups[cm_task] = []
        task_groups[cm_task].append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_original_task": trajectory_original_task,
                "final_reward": final_reward,
                "is_matching": cm_task == trajectory_original_task,
            }
        )

        # Track trajectory rewards across all tasks
        if trajectory_id not in trajectory_rewards:
            trajectory_rewards[trajectory_id] = {}
        trajectory_rewards[trajectory_id][cm_task] = final_reward

    if not task_groups:
        return {"error": "No valid confusion matrix data found"}

    # Calculate task discrimination metrics
    task_discrimination = {}
    overall_discrimination_scores = []

    for task in task_groups:
        matching_rewards = [r["final_reward"] for r in task_groups[task] if r["is_matching"]]
        non_matching_rewards = [r["final_reward"] for r in task_groups[task] if not r["is_matching"]]

        if matching_rewards and non_matching_rewards:
            avg_matching = np.mean(matching_rewards)
            avg_non_matching = np.mean(non_matching_rewards)
            discrimination_score = avg_matching - avg_non_matching

            task_discrimination[task] = {
                "avg_matching_reward": avg_matching,
                "avg_non_matching_reward": avg_non_matching,
                "discrimination_score": discrimination_score,
                "num_matching": len(matching_rewards),
                "num_non_matching": len(non_matching_rewards),
            }
            overall_discrimination_scores.append(discrimination_score)

    # Overall metrics
    overall_metrics = {}
    if overall_discrimination_scores:
        overall_metrics = {
            "mean_discrimination_score": np.mean(overall_discrimination_scores),
            "std_discrimination_score": np.std(overall_discrimination_scores),
            "min_discrimination_score": np.min(overall_discrimination_scores),
            "max_discrimination_score": np.max(overall_discrimination_scores),
        }

    return {
        "num_tasks": len(task_groups),
        "num_trajectories": len(trajectory_rewards),
        "num_samples": len(results),
        "task_discrimination": task_discrimination,
        "overall_metrics": overall_metrics,
    }


def run_policy_ranking_eval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run policy_ranking evaluation analysis."""
    # Group results by task
    task_groups = {}

    for r in results:
        # Get task and quality label from metadata
        meta = r.get("metadata", {}) or {}
        task = meta.get("task")
        quality_label = meta.get("quality_label")

        if task is None or quality_label is None:
            continue

        # Only consider failure, successful, suboptimal
        if quality_label not in ["failure", "successful", "suboptimal"]:
            continue

        # Get final reward (last value of progress_pred_A)
        progress_pred = r.get("progress_pred_A", [])
        if not progress_pred or len(progress_pred) == 0:
            continue

        final_reward = float(progress_pred[-1])

        # Group by task
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(
            {
                "quality_label": quality_label,
                "final_reward": final_reward,
            }
        )

    if not task_groups:
        return {"error": "No valid policy ranking data found"}

    # For each task, compute ranking correlation
    task_correlations = []
    task_details = {}

    for task, trajectories in task_groups.items():
        # Create gold ranking: failure=1, suboptimal=2, successful=3
        gold_ranks = []
        predicted_ranks = []

        for traj in trajectories:
            quality = traj["quality_label"]
            reward = traj["final_reward"]

            # Gold ranking
            if quality == "failure":
                gold_rank = 1
            elif quality == "suboptimal":
                gold_rank = 2
            elif quality == "successful":
                gold_rank = 3
            else:
                continue  # Skip unknown quality labels

            gold_ranks.append(gold_rank)
            predicted_ranks.append(reward)

        # Skip if we don't have at least 2 different quality levels
        unique_qualities = set(traj["quality_label"] for traj in trajectories)
        if len(unique_qualities) < 2:
            continue

        # Compute Spearman correlation
        if len(gold_ranks) >= 2:
            spearman_corr = compute_spearman(gold_ranks, predicted_ranks)
            if not np.isnan(spearman_corr):
                task_correlations.append(spearman_corr)
                task_details[task] = {
                    "spearman_correlation": spearman_corr,
                    "num_trajectories": len(trajectories),
                    "quality_distribution": {
                        q: sum(1 for t in trajectories if t["quality_label"] == q) for q in unique_qualities
                    },
                    "gold_ranks": gold_ranks,
                    "predicted_rewards": predicted_ranks,
                }

    # Overall metrics
    overall_metrics = {}
    if task_correlations:
        overall_metrics = {
            "mean_spearman_correlation": np.mean(task_correlations),
            "std_spearman_correlation": np.std(task_correlations),
            "min_spearman_correlation": np.min(task_correlations),
            "max_spearman_correlation": np.max(task_correlations),
            "num_tasks_with_valid_correlation": len(task_correlations),
        }

    return {
        "num_tasks": len(task_groups),
        "num_samples": len(results),
        "num_tasks_with_valid_correlation": len(task_correlations),
        "task_details": task_details,
        "overall_metrics": overall_metrics,
    }


def run_policy_ranking_eval_per_ranked_set(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run policy_ranking evaluation analysis per ranked set."""
    # Group results by task
    task_groups = {}

    for r in results:
        meta = r.get("metadata", {}) or {}
        task = meta.get("task")
        quality_label = meta.get("quality_label")

        if task is None or quality_label is None:
            continue

        if quality_label not in ["failure", "successful", "suboptimal"]:
            continue

        progress_pred = r.get("progress_pred_A", [])
        if not progress_pred or len(progress_pred) == 0:
            continue

        final_reward = float(progress_pred[-1])

        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(
            {
                "quality_label": quality_label,
                "final_reward": final_reward,
            }
        )

    if not task_groups:
        return {"error": "No valid policy ranking data found"}

    # For each task, compute ranking correlation
    task_correlations = []
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
        k_values = [3] if len(present_labels) == 3 else [2]

        spearman_corrs = []

        for k in k_values:
            for labels_combo in combinations(present_labels, k):
                gold_ranks = [quality_order[q] for q in labels_combo]
                # All ways to pick one reward per selected quality
                for rewards_tuple in product(*(quality_to_rewards[q] for q in labels_combo)):
                    spearman_corr = compute_spearman(gold_ranks, list(rewards_tuple))
                    if not np.isnan(spearman_corr):
                        spearman_corrs.append(spearman_corr)

        if spearman_corrs:
            avg_spearman_corr = np.mean(spearman_corrs)
            task_correlations.append(avg_spearman_corr)
            task_details[task] = {
                "average_spearman_correlation": avg_spearman_corr,
                "num_triplets/tuples": len(spearman_corrs),
                "quality_distribution": {
                    "failure": len(quality_to_rewards["failure"]),
                    "suboptimal": len(quality_to_rewards["suboptimal"]),
                    "successful": len(quality_to_rewards["successful"]),
                },
            }

    overall_metrics = {}
    if task_correlations:
        overall_metrics = {
            "mean_spearman_correlation": np.mean(task_correlations),
            "std_spearman_correlation": np.std(task_correlations),
            "min_spearman_correlation": np.min(task_correlations),
            "max_spearman_correlation": np.max(task_correlations),
            "num_tasks_with_valid_correlation": len(task_correlations),
        }

    return {
        "num_tasks": len(task_groups),
        "num_samples": len(results),
        "num_tasks_with_valid_correlation": len(task_correlations),
        "task_details": task_details,
        "overall_metrics": overall_metrics,
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
    parser.add_argument(
        "--eval_logs_dir", type=str, default=None, help="Root directory containing eval_logs structure"
    )
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to visualize")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
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
                    print(f"    Analyzing {eval_type}...")

                    try:
                        results = load_results(str(eval_file))
                        metrics = analyze_evaluation_type(eval_type, results)
                        dataset_results[eval_type] = metrics
                        print(f"      ✓ Completed {eval_type} analysis")
                    except Exception as e:
                        print(f"      ✗ Failed to analyze {eval_type}: {e}")
                        dataset_results[eval_type] = {"error": str(e)}

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

    # Directory mode: process known files with tailored analyses (legacy mode)
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
        sf_results = _load_if_exists("success_failure_preference.json")
        if sf_results:
            print("Running analyses for success_failure_preference.json:")
            metrics = run_success_failure_eval(sf_results)
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric_name}: {value:.4f}")
                else:
                    print(f"  - {metric_name}: {value}")

        # reward_alignment_progress.json: reuse compute_metrics and summaries
        ra_results = _load_if_exists("reward_alignment_progress.json")
        if ra_results:
            print("Running analyses for reward_alignment_progress.json:")
            metrics = run_reward_alignment_eval(ra_results)
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric_name}: {value:.4f}")
                else:
                    print(f"  - {metric_name}: {value}")
        else:
            print("No analyses run for reward_alignment_progress.json")

        # confusion_matrix.json: create confusion matrix analysis
        cm_results = _load_if_exists("confusion_matrix.json")
        if cm_results:
            print("Running analyses for confusion_matrix.json:")
            metrics = run_confusion_matrix_eval(cm_results)
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric_name}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  - {metric_name}:")
                    for sub_metric, sub_value in value.items():
                        if isinstance(sub_value, float):
                            print(f"    {sub_metric}: {sub_value:.4f}")
                        else:
                            print(f"    {sub_metric}: {sub_value}")
                else:
                    print(f"  - {metric_name}: {value}")
        else:
            print("No analyses run for confusion_matrix.json")

        # policy_ranking_progress.json: create policy ranking analysis
        pr_results = _load_if_exists("policy_ranking_progress.json")
        if pr_results:
            print("Running analyses for policy_ranking_progress.json:")
            metrics = run_policy_ranking_eval(pr_results)
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric_name}: {value:.4f}")
                else:
                    print(f"  - {metric_name}: {value}")
        else:
            print("No analyses run for policy_ranking_progress.json")

        print("Directory processing complete.")
        print("Done!")
        return


if __name__ == "__main__":
    main()
