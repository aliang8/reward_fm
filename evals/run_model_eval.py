#!/usr/bin/env python3
"""
Client script to iteratively generate evaluation batches and send them to an
evaluation server (e.g., localhost:8000). The server is expected to return a
dictionary with keys:
  - predictions: [] # list of predictions for each sample in the batch. (1 if chosen preferred, else 0, -1 means no preference)
  - reward_chosen: [] # list of list of per-frame rewards for the chosen trajectory
  - reward_rejected: [] # list of list of per-frame rewards for the rejected trajectory

Usage:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8000 --num_batches=10 --batch_size=4
  
  # Process entire dataset:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8000 --num_batches=-1 --batch_size=4
  
  # Override config values:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --set data.max_frames=16 --set data.eval_subset_size=1000
"""

from __future__ import annotations

import argparse
import ast
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.stats import spearmanr
from rich.table import Table
from rich.console import Console

from tqdm import tqdm

from rfm.configs.experiment_configs import ExperimentConfig
from rfm.utils.setup_utils import setup_eval_dataset
from evals.eval_utils import (
    load_experiment_config_from_yaml,
    build_batch_payload,
    post_batch,
)
from rfm.data.batch_collator import PreferenceSample, SimilaritySample

KEY_TO_MEANING = {
    "eval_loss": "Loss",
    "eval_accuracy": "Accuracy of Predicting the Correct Preference",
    "mse_progress_A": "MSE between progress_pred_A and target_progress_A",
    "mse_progress_B": "MSE between progress_pred_B and target_progress_B",
    "spearman_progress_A": "Spearman correlation between progress_pred_A and target_progress_A",
    "spearman_progress_B": "Spearman correlation between progress_pred_B and target_progress_B",
}


def _compute_metrics_from_response(
    response: Dict[str, Any], samples: List[Union[PreferenceSample, SimilaritySample]]
) -> Dict[str, Any]:
    """Compute metrics from model response and samples.

    This function efficiently computes metrics by:
    1. Computing all base metrics once for the entire batch
    2. Creating granular metrics by subsetting the base metrics for each category
    3. Avoiding redundant calculations for different granular breakdowns
    """
    # Extract predictions and progress predictions
    preds = response.get("predictions", [])
    progress_pred_A = response.get("progress_pred_A", [])
    progress_pred_B = response.get("progress_pred_B", [])
    target_progress_A = [s.target_progress_A[::2] for s in samples]
    target_progress_B = [s.target_progress_B[::2] for s in samples]

    # Compute ALL base metrics once for efficiency
    base_metrics = {}

    # Preference accuracy
    if any(s.sample_type == "preference" for s in samples):
        correct_preds = sum(1 for p in preds if p == 1)  # 1 means chosen trajectory is preferred
        base_metrics["eval_accuracy"] = correct_preds / len(preds)

    # Progress metrics: compute MSE and Spearman for A and B
    def _compute_mse_and_spearman(pred_list, target_list):
        # Align lengths and compute metrics per-sample, then average
        mses = []
        spearmans = []
        for pred, target in zip(pred_list, target_list):
            if pred is None or target is None:
                continue
            try:
                pred_arr = np.array(pred, dtype=float)
                target_arr = np.array(target, dtype=float)
                # Align lengths by truncation to the shorter length
                L = min(len(pred_arr), len(target_arr))
                if L == 0:
                    continue
                pred_arr = pred_arr[:L]
                target_arr = target_arr[:L]
                mse = float(np.mean((pred_arr - target_arr) ** 2))
                mses.append(mse)
                # Spearman correlation
                corr, _ = spearmanr(pred_arr, target_arr)
                if corr is not None and not np.isnan(corr):
                    spearmans.append(float(corr))
            except Exception as e:
                # Skip malformed items
                continue
        avg_mse = float(np.mean(mses)) if mses else None
        avg_spear = float(np.mean(spearmans)) if spearmans else None
        return avg_mse, avg_spear

    mse_A, spear_A = _compute_mse_and_spearman(progress_pred_A, target_progress_A)
    mse_B, spear_B = _compute_mse_and_spearman(progress_pred_B, target_progress_B)
    base_metrics["mse_progress_A"] = mse_A
    base_metrics["mse_progress_B"] = mse_B
    base_metrics["spearman_progress_A"] = spear_A
    base_metrics["spearman_progress_B"] = spear_B

    # No global progress alignment computed here; per-sample Spearman handled above for A and B.

    # Now create granular metrics by subsetting the base metrics
    granular_metrics = {}

    # Helper function to subset metrics for a specific category
    def get_subset_metrics(indices: List[int], metric_prefix: str) -> Dict[str, Optional[float]]:
        """Get subset metrics for a specific category by filtering base metrics."""
        if not indices:
            return {}

        subset_metrics = {}

        # Filter predictions for this subset
        subset_preds = [preds[i] for i in indices if i < len(preds)]
        subset_progress_pred_A = [progress_pred_A[i] for i in indices if i < len(progress_pred_A)]
        subset_progress_pred_B = [progress_pred_B[i] for i in indices if i < len(progress_pred_B)]
        subset_target_A = [target_progress_A[i] for i in indices if i < len(target_progress_A)]
        subset_target_B = [target_progress_B[i] for i in indices if i < len(target_progress_B)]

        if not subset_preds:
            return {}

        # Compute accuracy for this subset
        correct = sum(1 for p in subset_preds if p == 1)  # 1 means chosen trajectory is preferred
        subset_metrics[f"accuracy_{metric_prefix}"] = correct / len(subset_preds)

        # Compute progress metrics for subset (A and B)
        subset_mse_A, subset_spear_A = _compute_mse_and_spearman(subset_progress_pred_A, subset_target_A)
        subset_mse_B, subset_spear_B = _compute_mse_and_spearman(subset_progress_pred_B, subset_target_B)
        subset_metrics[f"mse_progress_A_{metric_prefix}"] = subset_mse_A
        subset_metrics[f"mse_progress_B_{metric_prefix}"] = subset_mse_B
        subset_metrics[f"spearman_progress_A_{metric_prefix}"] = subset_spear_A
        subset_metrics[f"spearman_progress_B_{metric_prefix}"] = subset_spear_B

        return subset_metrics

    # Data gen strategy analysis
    data_gen_strategy_indices = {}
    for i, sample in enumerate(samples):
        data_gen_strategy = sample.data_gen_strategy
        if data_gen_strategy not in data_gen_strategy_indices:
            data_gen_strategy_indices[data_gen_strategy] = []
        data_gen_strategy_indices[data_gen_strategy].append(i)

    # Add data gen strategy counts
    for data_gen_strategy, indices in data_gen_strategy_indices.items():
        granular_metrics[f"count_{data_gen_strategy}"] = len(indices)

    # Get subset metrics for each data gen strategy
    for data_gen_strategy, indices in data_gen_strategy_indices.items():
        subset_metrics = get_subset_metrics(indices, data_gen_strategy)
        granular_metrics.update(subset_metrics)

    # Rewound frame analysis (for paired video samples)
    rewound_indices = [
        i for i, s in enumerate(samples) if hasattr(s, "num_frames_rewound") and s.num_frames_rewound is not None
    ]
    if rewound_indices:
        rewound_frame_counts = {}
        for i in rewound_indices:
            num_frames = samples[i].num_frames_rewound
            if num_frames not in rewound_frame_counts:
                rewound_frame_counts[num_frames] = []
            rewound_frame_counts[num_frames].append(i)

        # Add rewound frame counts
        for num_frames, indices in rewound_frame_counts.items():
            granular_metrics[f"count_rewound_{num_frames}"] = len(indices)

        # Get subset metrics for each rewound frame category
        for num_frames, indices in rewound_frame_counts.items():
            subset_metrics = get_subset_metrics(indices, f"rewound_{num_frames}")
            granular_metrics.update(subset_metrics)

    # Rejected quality analysis (for paired video samples)
    rejected_quality_indices = {}
    for i, sample in enumerate(samples):
        if hasattr(sample, "rejected_quality_label") and sample.rejected_quality_label:
            quality = sample.rejected_quality_label
            if quality not in rejected_quality_indices:
                rejected_quality_indices[quality] = []
            rejected_quality_indices[quality].append(i)

    # Add rejected quality counts
    for quality, indices in rejected_quality_indices.items():
        granular_metrics[f"count_rejected_{quality}"] = len(indices)

    # Get subset metrics for each rejected quality category
    for quality, indices in rejected_quality_indices.items():
        subset_metrics = get_subset_metrics(indices, f"rejected_{quality}")
        granular_metrics.update(subset_metrics)

    return {
        **base_metrics,  # Include base metrics
        **granular_metrics,  # Include all granular metrics
    }


def _evaluate_samples(server_url: str, samples: List[Any]) -> Dict[str, Any]:
    payload = build_batch_payload(samples)
    resp = post_batch(server_url, payload)
    return _compute_metrics_from_response(resp, samples)


def iter_eval_batches(
    cfg: ExperimentConfig,
    server_url: str,
    num_batches: int = 10,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    # Create eval data generator and dataset-like iterator
    dataset = setup_eval_dataset(cfg)

    # Determine actual number of batches
    dataset_size = len(dataset)
    if num_batches == -1:
        # Go through the full dataset
        actual_num_batches = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
        print(f"\nProcessing FULL DATASET: {dataset_size} samples in {actual_num_batches} batches of size {batch_size}")
    else:
        actual_num_batches = num_batches
        print(f"\nProcessing {actual_num_batches} batches of size {batch_size} (dataset size: {dataset_size})")

    results: List[Dict[str, Any]] = []
    idx = 0
    batch_idx = 0

    # for batch_idx in tqdm(range(actual_num_batches), desc=f"Evaluating batch [{batch_idx+1}/{actual_num_batches}]"):

    for batch_idx in range(actual_num_batches):
        # Check if we've reached the end of the dataset
        if idx >= dataset_size:
            print(f"\nReached end of dataset after {batch_idx} batches")
            break

        # Assemble a batch of Sample objects (Preference or Similarity)
        batch_samples = []
        for j in range(batch_size):
            if idx + j < dataset_size:
                batch_samples.append(dataset[idx + j])
            else:
                break  # Don't go beyond dataset size

        if not batch_samples:
            break  # No more samples

        # Evaluate this batch
        batch_result = _evaluate_samples(server_url, batch_samples)
        results.append(batch_result)

        # Print batch results immediately
        print(
            f"\n" + "=" * 80,
        )
        print(f"BATCH {batch_idx + 1}/{actual_num_batches} RESULTS (Processed)")
        print(f"   Progress: {idx}/{dataset_size} samples ({idx / dataset_size * 100:.1f}%)")
        print("=" * 80)

        # Extract main metrics for this batch
        keys = [
            "eval_accuracy",
            "mse_progress_A",
            "mse_progress_B",
            "spearman_progress_A",
            "spearman_progress_B",
        ]
        batch_main_metrics = {k: batch_result.get(k) for k in keys if k in batch_result}

        # Extract granular metrics for this batch
        granular_keys = [
            k
            for k in batch_result.keys()
            if k.startswith(
                (
                    "accuracy_",
                    "mse_progress_A_",
                    "mse_progress_B_",
                    "spearman_progress_A_",
                    "spearman_progress_B_",
                    "count_",
                )
            )
        ]
        batch_granular = {k: batch_result.get(k) for k in granular_keys if k in batch_result}

        # Print batch summary
        generate_metrics_summary(
            metrics=batch_main_metrics,
            granular_metrics=batch_granular,
            title=f"BATCH {batch_idx + 1}/{actual_num_batches} RESULTS",
            level="batch",
        )

        # Add separator between batches (except for the last one)
        if batch_idx < actual_num_batches - 1:
            print("\n" + "─" * 80)
            print("NEXT BATCH")
            print("─" * 80)

        import sys

        sys.stdout.flush()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/config.yaml")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to evaluate. Use -1 to process the entire dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dot-path assignments, e.g., --set data.max_frames=8 --set model.base_model_id='Qwen/...'.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config_from_yaml(args.config_path)

    # Apply overrides from --set key=value (dot-path)
    for assignment in args.set:
        if "=" not in assignment:
            print(f"Warning: Invalid --set argument '{assignment}', skipping. Use format: key=value")
            continue
        key, value_str = assignment.split("=", 1)
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not hasattr(target, p):
                print(f"Warning: Config path '{key}' is invalid, skipping override")
                break
            target = getattr(target, p)
        else:
            setattr(target, parts[-1], value)
            print(f"Applied config override: {key} = {value}")

    results = iter_eval_batches(
        cfg=cfg,
        server_url=args.server_url,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
    )

    # Print an aggregated summary
    keys = [
        "eval_accuracy",
        "mse_progress_A",
        "mse_progress_B",
        "spearman_progress_A",
        "spearman_progress_B",
    ]

    # Add granular metrics keys
    granular_keys = []
    for r in results:
        for k in r.keys():
            if k.startswith(
                (
                    "accuracy_",
                    "mse_progress_A_",
                    "mse_progress_B_",
                    "spearman_progress_A_",
                    "spearman_progress_B_",
                    "count_",
                )
            ):
                if k not in granular_keys:
                    granular_keys.append(k)

    # Sort granular keys for consistent display
    granular_keys.sort()

    agg = {k: 0.0 for k in keys + granular_keys}
    # keeps track of which keys are always None
    none_keys = {k: [] for k in keys + granular_keys}
    # keeps track of how many batches actually have each metric
    metric_counts = {k: 0 for k in keys + granular_keys}
    n = max(1, len(results))
    for r in results:
        for k in keys + granular_keys:
            if k in r and r[k] is None:
                none_keys[k].append(1)
            elif k in r and isinstance(r[k], (int, float)):
                none_keys[k].append(0)
                agg[k] += float(r[k])
                metric_counts[k] += 1

    # Only average metrics that actually appeared in some batches
    for k in keys + granular_keys:
        if metric_counts[k] > 0:
            agg[k] /= metric_counts[k]
        else:
            agg[k] = None  # Set to None if metric never appeared

    for k in none_keys:
        if all(none_keys[k]):
            print(f"WARNING: {k} is always None, removing")
            agg.pop(k)
            continue

    # Print evaluation summary (averaged across batches)
    # Create granular metrics dictionary from aggregated results
    granular_metrics = {k: agg[k] for k in granular_keys if k in agg}

    generate_metrics_summary(
        metrics=agg,
        granular_metrics=granular_metrics,
        title="EVALUATION SUMMARY (Averaged Across Batches)",
        level="final",
    )


def generate_metrics_summary(metrics: dict, granular_metrics: dict, title: str, level: str = "batch"):
    """Generate a nicely formatted metrics summary print using tables.

    Args:
        metrics: Dictionary of main metrics
        granular_metrics: Dictionary of granular metrics
        title: Title for the summary section
        level: Either "batch" or "final" to indicate the level of detail
    """
    print(f"\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Print main metrics
    if metrics:
        print(f"\nMain Metrics")
        print("-" * 12)

        # Create table for main metrics
        console = Console()
        main_table = Table(show_header=True, header_style="bold")
        main_table.add_column("Metric", style="cyan", no_wrap=True)
        main_table.add_column("Value", style="green")
        main_table.add_column("Description", style="yellow")

        for k in sorted(metrics.keys()):
            if metrics[k] is not None:
                value = metrics[k]
                # Round to 2 decimal places
                if isinstance(value, (int, float)):
                    value = round(value, 2)

                # Format the metric name for better readability
                display_name = k.replace("_", " ").title()

                # Add description based on metric type
                if k.startswith("accuracy"):
                    description = "Accuracy of Predicting the Correct Preference"
                elif k.startswith("avg_reward_chosen"):
                    description = "Average Reward Assigned to (Chosen)"
                elif k.startswith("avg_reward_rejected"):
                    description = "Average Reward (Rejected)"
                elif k.startswith("reward_diff"):
                    description = "Reward Difference between Chosen and Rejected"
                elif k.startswith("progress_alignment"):
                    description = "Spearman Correlation between Predicted Progress and Ground Truth Progress"
                else:
                    description = k

                main_table.add_row(display_name, str(value), description)
            else:
                # Handle None values
                display_name = k.replace("_", " ").title()
                main_table.add_row(
                    display_name, "Not Available", f"This metric was not computed for any samples in this {level}"
                )

        console.print(main_table)

    # Print granular metrics if available
    if granular_metrics:
        print(f"\n" + "=" * 80)
        print(f"GRANULAR METRICS ANALYSIS ({level.upper()} LEVEL)")
        print("=" * 80)

        # Separate count metrics from performance metrics
        count_metrics = {k: v for k, v in granular_metrics.items() if k.startswith("count_")}
        performance_metrics = {k: v for k, v in granular_metrics.items() if not k.startswith("count_")}

        # Print count metrics first
        if count_metrics:
            total_samples = sum(count_metrics.values())
            print(f"\nSample Counts (Total: {total_samples})")
            print("-" * 15)

            count_table = Table(show_header=True, header_style="bold")
            count_table.add_column("Category", style="cyan", no_wrap=True)
            count_table.add_column("Count", style="green")
            count_table.add_column("Description", style="yellow")

            for k in sorted(count_metrics.keys()):
                if count_metrics[k] is not None:
                    value = count_metrics[k]
                    # Format the metric name for better readability
                    display_name = k.replace("count_", "").replace("_", " ").title()
                    count_table.add_row(display_name, str(value), "Number of samples in this category")

            console.print(count_table)

        # Group performance metrics by type for better organization
        metric_groups = {
            "Data Gen Strategy Analysis": [],
            "Rewound Frame Analysis": [],
            "Rejected Quality Analysis": [],
        }

        for k in performance_metrics.keys():
            if k.startswith("accuracy_") and not k.startswith(("accuracy_rewound_", "accuracy_rejected_")):
                metric_groups["Data Gen Strategy Analysis"].append(k)
            elif k.startswith("accuracy_rewound_"):
                metric_groups["Rewound Frame Analysis"].append(k)
            elif k.startswith("accuracy_rejected_"):
                metric_groups["Rejected Quality Analysis"].append(k)
            elif k.startswith(("mse_progress_A_", "mse_progress_B_", "spearman_progress_A_", "spearman_progress_B_")):
                # Add to appropriate group based on the metric type
                if "rewound_" in k:
                    metric_groups["Rewound Frame Analysis"].append(k)
                elif "rejected_" in k:
                    metric_groups["Rejected Quality Analysis"].append(k)
                else:
                    metric_groups["Data Gen Strategy Analysis"].append(k)

        # Print each group with nice formatting
        for group_name, group_metrics in metric_groups.items():
            if group_metrics:
                print(f"\n{group_name}")
                print("-" * len(group_name))

                # Create table for this group
                group_table = Table(show_header=True, header_style="bold")
                group_table.add_column("Metric", style="cyan", no_wrap=True)
                group_table.add_column("Value", style="green")
                group_table.add_column("Description", style="yellow")

                # Sort metrics within each group for consistent display
                sorted_metrics = sorted(group_metrics)

                for k in sorted_metrics:
                    if k in performance_metrics and performance_metrics[k] is not None:
                        value = performance_metrics[k]
                        # Round to 2 decimal places
                        if isinstance(value, (int, float)):
                            value = round(value, 2)

                        # Format the metric name for better readability
                        display_name = k.replace("_", " ").title()

                        # Add description for granular metrics
                        if k.startswith("accuracy_"):
                            description = "Accuracy of Predicting the Correct Preference (granular breakdown)"
                        elif k.startswith("mse_progress_A_"):
                            description = "MSE between progress_pred_A and target_progress_A (granular)"
                        elif k.startswith("mse_progress_B_"):
                            description = "MSE between progress_pred_B and target_progress_B (granular)"
                        elif k.startswith("spearman_progress_A_"):
                            description = "Spearman correlation between progress_pred_A and target_progress_A (granular)"
                        elif k.startswith("spearman_progress_B_"):
                            description = "Spearman correlation between progress_pred_B and target_progress_B (granular)"
                        else:
                            description = k

                        group_table.add_row(display_name, str(value), description)
                    elif k in performance_metrics and performance_metrics[k] is None:
                        # Handle None values in granular metrics
                        display_name = k.replace("_", " ").title()
                        group_table.add_row(
                            display_name,
                            "Not Available",
                            "This metric was not computed for any samples in this category",
                        )

                console.print(group_table)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
