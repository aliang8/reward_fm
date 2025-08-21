#!/usr/bin/env python3
"""
Script to compile evaluation results from JSON files, compute metrics, and create visualizations.

Usage:
    # Run with default config (will look for results in log_dir/model_name/dataset_name/results.json):
    # All output files (metrics.json, visualizations) will be saved in the same directory
    python evals/compile_results.py
    
    # Run with custom config:
    python evals/compile_results.py --config rfm/configs/eval_config.yaml
    
    # Run with custom max samples for visualization:
    python evals/compile_results.py --max_samples 10
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


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive metrics from results."""
    metrics = {}

    # Preference accuracy
    correct_predictions = 0
    total_predictions = 0

    for result in results:
        if result.get("predicted_preference") is not None and result.get("preference_label") is not None:
            if result["predicted_preference"] == result["preference_label"]:
                correct_predictions += 1
            total_predictions += 1

    if total_predictions > 0:
        metrics["preference_accuracy"] = correct_predictions / total_predictions
    else:
        metrics["preference_accuracy"] = None

    # Progress metrics
    mse_progress_chosen = []
    mse_progress_rejected = []
    spearman_progress_chosen = []
    spearman_progress_rejected = []

    for result in results:
        # Chosen trajectory progress
        pred_chosen = np.array(result["progress_pred_chosen"])
        target_chosen = np.array(result["target_progress_chosen"][::2])

        # Compute MSE
        mse_chosen = np.mean((pred_chosen - target_chosen) ** 2)
        mse_progress_chosen.append(mse_chosen)

        # Compute Spearman correlation
        corr_chosen, _ = spearmanr(pred_chosen, target_chosen)
        if not np.isnan(corr_chosen):
            spearman_progress_chosen.append(corr_chosen)

        if "progress_pred_rejected" in result:
            # Rejected trajectory progress
            pred_rejected = np.array(result["progress_pred_rejected"])
            target_rejected = np.array(result["target_progress_rejected"][::2])

            # Compute MSE
            mse_rejected = np.mean((pred_rejected - target_rejected) ** 2)
            mse_progress_rejected.append(mse_rejected)

            # Compute Spearman correlation
            corr_rejected, _ = spearmanr(pred_rejected, target_rejected)
            spearman_progress_rejected.append(corr_rejected)
        else:
            mse_progress_rejected.append(None)
            spearman_progress_rejected.append(None)

    mse_progress_chosen = [mse for mse in mse_progress_chosen if mse is not None]
    mse_progress_rejected = [mse for mse in mse_progress_rejected if mse is not None]
    spearman_progress_chosen = [corr for corr in spearman_progress_chosen if corr is not None]
    spearman_progress_rejected = [corr for corr in spearman_progress_rejected if corr is not None]

    metrics["mse_progress_chosen"] = np.mean(mse_progress_chosen)
    metrics["mse_progress_rejected"] = np.mean(mse_progress_rejected)
    metrics["spearman_progress_chosen"] = np.mean(spearman_progress_chosen)
    metrics["spearman_progress_rejected"] = np.mean(spearman_progress_rejected)
    metrics["mse_progress_chosen_std"] = np.std(mse_progress_chosen)
    metrics["mse_progress_rejected_std"] = np.std(mse_progress_rejected)
    metrics["spearman_progress_chosen_std"] = np.std(spearman_progress_chosen)
    metrics["spearman_progress_rejected_std"] = np.std(spearman_progress_rejected)
    
    # Sample counts by data generation strategy
    strategy_counts = {}
    for result in results:
        strategy = result["data_gen_strategy"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    metrics["strategy_counts"] = strategy_counts

    # Rewound frame analysis
    rewound_counts = {}
    for result in results:
        rewound = result["num_frames_rewound"]
        if rewound is not None:
            rewound_counts[str(rewound)] = rewound_counts.get(str(rewound), 0) + 1

    metrics["rewound_counts"] = rewound_counts

    return metrics


def create_visualizations(results: List[Dict[str, Any]], output_dir: Path, max_samples: int = 5):
    """Create visualizations for selected samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Randomly select samples for visualization
    if len(results) <= max_samples:
        selected_results = results
    else:
        selected_indices = np.random.choice(len(results), max_samples, replace=False)
        selected_results = [results[i] for i in selected_indices]

    # Create progress plots
    _create_progress_plots(selected_results, video_dir)


def _create_progress_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Create progress prediction plots for selected samples."""
    for i, result in enumerate(results):
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Chosen trajectory progress
        ax1.set_title(f"Sample {i + 1}: Chosen Trajectory Progress", fontsize=12, weight="bold")
        if result.get("progress_pred_chosen") and result.get("target_progress_chosen"):
            pred_chosen = result["progress_pred_chosen"]
            target_chosen = result["target_progress_chosen"]

            x_pred = list(range(len(pred_chosen))) 
            x_pred = [x * 2 for x in x_pred]
            x_target = list(range(len(target_chosen)))

            # Use scatter plots since progress is predicted at every other frame
            ax1.scatter(x_pred, pred_chosen, c="blue", s=50, label="Predicted Progress", alpha=0.7, edgecolors="black")
            ax1.scatter(x_target, target_chosen, c="red", s=50, label="Target Progress", alpha=0.7, edgecolors="black", marker="s")

            ax1.set_xlabel("Frame")
            ax1.set_ylabel("Progress")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_xticks(x_pred)
        else:
            ax1.text(
                0.5,
                0.5,
                "No progress data available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=12,
                color="red",
            )

        # Plot 2: Rejected trajectory progress
        ax2.set_title(f"Sample {i + 1}: Rejected Trajectory Progress", fontsize=12, weight="bold")
        if result.get("progress_pred_rejected") and result.get("target_progress_rejected"):
            pred_rejected = result["progress_pred_rejected"]
            target_rejected = result["target_progress_rejected"]

            x_pred = list(range(len(pred_rejected)))
            x_pred = [x * 2 for x in x_pred]
            x_target = list(range(len(target_rejected)))

            # Use scatter plots since progress is predicted at every other frame
            ax2.scatter(x_pred, pred_rejected, c="green", s=50, label="Predicted Progress", alpha=0.7, edgecolors="black")
            ax2.scatter(x_target, target_rejected, c="red", s=50, label="Target Progress", alpha=0.7, edgecolors="black", marker="s")

            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Progress")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.set_xticks(x_pred)
        else:
            ax2.text(
                0.5,
                0.5,
                "No progress data available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
                color="red",
            )

        # Add sample info
        info_text = f"Chosen Task: {result.get('chosen_task', 'unknown')}\n"
        info_text += f"Rejected Task: {result.get('rejected_task', 'unknown')}\n"
        info_text += f"Data Gen Strategy: {result.get('data_gen_strategy', 'unknown')}\n"
        info_text += f"Rewound Frames: {result.get('num_frames_rewound', 'N/A')}\n"
        info_text += f"Preference Label: {result.get('preference_label', 'N/A')}\n"
        info_text += f"Predicted Preference: {result.get('predicted_preference', 'N/A')}"

        fig.suptitle(f"Sample {i + 1} Progress Predictions", fontsize=14, weight="bold")
        fig.text(
            0.02,
            -0.2,
            info_text,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i + 1}_progress.png", dpi=150, bbox_inches="tight")
        plt.close()


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a nicely formatted metrics summary."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    print(f"\nPreference Accuracy: {metrics.get('preference_accuracy', 'N/A'):.4f}")

    print(f"\nProgress Metrics (Chosen Trajectory):")
    print(f"  MSE: {metrics.get('mse_progress_chosen', 'N/A'):.6f} ± {metrics.get('mse_progress_chosen_std', 'N/A'):.6f}")
    print(
        f"  Spearman Correlation: {metrics.get('spearman_progress_chosen', 'N/A'):.4f} ± {metrics.get('spearman_progress_chosen_std', 'N/A'):.4f}"
    )

    print(f"\nProgress Metrics (Rejected Trajectory):")
    print(f"  MSE: {metrics.get('mse_progress_rejected', 'N/A'):.6f} ± {metrics.get('mse_progress_rejected_std', 'N/A'):.6f}")
    print(
        f"  Spearman Correlation: {metrics.get('spearman_progress_rejected', 'N/A'):.4f} ± {metrics.get('spearman_progress_rejected_std', 'N/A'):.4f}"
    )

    print(f"\nData Generation Strategy Counts:")
    for strategy, count in metrics.get("strategy_counts", {}).items():
        print(f"  {strategy}: {count}")

    print(f"\nRewound Frame Counts:")
    for rewound, count in metrics.get("rewound_counts", {}).items():
        print(f"  {rewound} frames: {count}")

    print("\n" + "=" * 80)


def main():
    import yaml
    from rfm.configs.experiment_configs import DataConfig
    from rfm.configs.eval_configs import EvaluationConfig
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compile evaluation results and create visualizations")
    parser.add_argument(
        "--config", type=str, default="rfm/configs/eval_config.yaml", help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5, help="Maximum number of samples to visualize"
    )
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    # Load results - handle the new directory structure
    model_name = cfg.model_path.replace("/", "_")
    dataset_name = f"{cfg.data.eval_datasets[0]}_{cfg.data.eval_subsets[0]}"
    results_file = Path(cfg.log_dir) / model_name / dataset_name / "results.json"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please check the path and ensure the evaluation has been run first.")
        return
    
    results = load_results(str(results_file))
    print(f"Loaded {len(results)} samples from {results_file}")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # Print metrics summary
    print_metrics_summary(metrics)

    # Create visualizations in the same directory as results.json
    results_dir = results_file.parent
    print(f"Creating visualizations in: {results_dir}")
    create_visualizations(results, results_dir, args.max_samples)

    # Save metrics to file in the same directory as results.json
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    print("Done!")


if __name__ == "__main__":
    main()
