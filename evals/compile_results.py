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
from rfm.utils.video_utils import extract_frames_from_video


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

        if result.get("progress_pred_rejected") is not None and result.get("target_progress_rejected") is not None:
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

    # Bin delta accuracy analysis
    delta_accuracy = analyze_bin_delta_accuracy(results)
    if delta_accuracy:
        metrics["bin_delta_accuracy"] = delta_accuracy

        # Add summary statistics
        total_correct = sum(delta_accuracy[delta]["correct"] for delta in delta_accuracy)
        total_incorrect = sum(delta_accuracy[delta]["incorrect"] for delta in delta_accuracy)
        total_abstained = sum(delta_accuracy[delta]["abstained"] for delta in delta_accuracy)
        total_samples = sum(delta_accuracy[delta]["total"] for delta in delta_accuracy)
        overall_delta_accuracy = total_correct / total_samples if total_samples > 0 else 0
        overall_abstention_rate = total_abstained / total_samples if total_samples > 0 else 0

        metrics["bin_delta_overall_accuracy"] = overall_delta_accuracy
        metrics["bin_delta_overall_abstention_rate"] = overall_abstention_rate
        metrics["bin_delta_total_samples"] = total_samples
        metrics["bin_delta_total_correct"] = total_correct
        metrics["bin_delta_total_incorrect"] = total_incorrect
        metrics["bin_delta_total_abstained"] = total_abstained

    return metrics


def analyze_bin_delta_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze preference accuracy based on the frame delta between bins.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with delta analysis results including abstentions
    """
    # Filter results that have bin indices
    bin_results = [r for r in results if "bin_idx_chosen" in r and "bin_idx_rejected" in r]

    if not bin_results:
        print("No results with bin indices found for delta analysis")
        return {}

    # Calculate frame delta for each sample
    delta_accuracy = {}

    for result in bin_results:
        bin_chosen = result["bin_idx_chosen"]
        bin_rejected = result["bin_idx_rejected"]

        # Calculate absolute frame delta
        frame_delta = abs(bin_chosen - bin_rejected)

        # Initialize delta entry if not exists
        if frame_delta not in delta_accuracy:
            delta_accuracy[frame_delta] = {
                "correct": 0,
                "incorrect": 0,
                "abstained": 0,
                "total": 0,
                "accuracy": 0.0,
                "abstention_rate": 0.0,
            }

        # Get prediction probability
        pred_prob = result.get("predicted_preference_prob", 0.5)

        # Determine if model abstained (probability close to 0.5)
        if abs(pred_prob - 0.5) <= 0.1:
            delta_accuracy[frame_delta]["abstained"] += 1
        else:
            # Count correct vs incorrect predictions
            if result["predicted_preference"] == result["preference_label"]:
                delta_accuracy[frame_delta]["correct"] += 1
            else:
                delta_accuracy[frame_delta]["incorrect"] += 1

        delta_accuracy[frame_delta]["total"] += 1

    # Calculate accuracy and abstention rate for each delta
    for delta in delta_accuracy:
        data = delta_accuracy[delta]
        data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        data["abstention_rate"] = data["abstained"] / data["total"] if data["total"] > 0 else 0.0

    return delta_accuracy


def create_bin_delta_plot(delta_accuracy: Dict[str, Any], output_dir: Path):
    """
    Create a stacked bar plot showing preference accuracy breakdown by frame delta.

    Args:
        delta_accuracy: Dictionary with delta analysis results
        output_dir: Directory to save the plot
    """
    if not delta_accuracy:
        print("No delta accuracy data to plot")
        return

    # Sort deltas for proper x-axis ordering
    deltas = sorted(delta_accuracy.keys())

    # Extract data for stacked bars (convert to percentages)
    correct_pcts = [delta_accuracy[delta]["correct"] / delta_accuracy[delta]["total"] * 100 for delta in deltas]
    abstained_pcts = [delta_accuracy[delta]["abstained"] / delta_accuracy[delta]["total"] * 100 for delta in deltas]
    incorrect_pcts = [delta_accuracy[delta]["incorrect"] / delta_accuracy[delta]["total"] * 100 for delta in deltas]
    total_counts = [delta_accuracy[delta]["total"] for delta in deltas]

    # Create the plot (single panel now)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create stacked bar chart showing percentage breakdown
    x = np.arange(len(deltas))
    width = 0.8

    # Create stacked bars with percentages
    bars1 = ax.bar(x, correct_pcts, width, label="Correct", color="green", alpha=0.7)
    bars2 = ax.bar(x, abstained_pcts, width, bottom=correct_pcts, label="Abstained", color="blue", alpha=0.7)
    bars3 = ax.bar(
        x,
        incorrect_pcts,
        width,
        bottom=np.array(correct_pcts) + np.array(abstained_pcts),
        label="Incorrect",
        color="red",
        alpha=0.7,
    )

    ax.set_xlabel("Frame Delta (|bin_chosen - bin_rejected|)")
    ax.set_ylabel("Preference Accuracy (%)")
    ax.set_title(
        "Preference Accuracy by Frame Delta Between Bins (Percentages)", fontsize=14, weight="bold", y=1.1, pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(deltas)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (correct_pct, abstained_pct, incorrect_pct) in enumerate(zip(correct_pcts, abstained_pcts, incorrect_pcts)):
        # Label for correct (bottom)
        if correct_pct > 0:
            ax.text(
                i, correct_pct / 2, f"{correct_pct:.1f}%", ha="center", va="center", fontweight="bold", color="white"
            )

        # Label for abstained (middle)
        if abstained_pct > 0:
            ax.text(
                i,
                correct_pct + abstained_pct / 2,
                f"{abstained_pct:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )

        # Label for incorrect (top)
        if incorrect_pct > 0:
            ax.text(
                i,
                correct_pct + abstained_pct + incorrect_pct / 2,
                f"{incorrect_pct:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )

    # Add sample count annotations above bars
    for i, count in enumerate(total_counts):
        ax.text(i, 102, f"n={count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "bin_delta_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Bin delta accuracy plot saved to: {output_dir / 'bin_delta_accuracy.png'}")


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
    _trajectory_progress_plot(selected_results, video_dir)
    
    # Create video binned frame visualizations if available
    _video_binned_frame_plot(selected_results, video_dir)


def _trajectory_progress_plot(results: List[Dict[str, Any]], output_dir: Path):
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
            ax1.scatter(
                x_target,
                target_chosen,
                c="red",
                s=50,
                label="Target Progress",
                alpha=0.7,
                edgecolors="black",
                marker="s",
            )

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
            ax2.scatter(
                x_pred, pred_rejected, c="green", s=50, label="Predicted Progress", alpha=0.7, edgecolors="black"
            )
            ax2.scatter(
                x_target,
                target_rejected,
                c="red",
                s=50,
                label="Target Progress",
                alpha=0.7,
                edgecolors="black",
                marker="s",
            )

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


def _video_binned_frame_plot(results: List[Dict[str, Any]], output_dir: Path):
    """Create frame sequence visualizations for video binned samples."""
    # Filter for video binned samples
    video_binned_results = [r for r in results if r.get("chosen_start_end") and r.get("rejected_start_end")]
    
    if not video_binned_results:
        print("No video binned samples found for frame visualization")
        return
    
    print(f"Creating frame visualizations for {len(video_binned_results)} video binned samples...")
    
    for i, result in enumerate(video_binned_results):
        if not all(key in result for key in ["video_path", "chosen_start_end", "rejected_start_end", "fps"]):
            print(f"Skipping sample {i+1}: missing video metadata")
            continue
            
        # Extract frames for chosen and rejected sequences
        all_frames = extract_frames_from_video(result["video_path"], result["fps"])
        
        # Slice frames based on start_end indices
        start_chosen, end_chosen = result["chosen_start_end"]
        start_rejected, end_rejected = result["rejected_start_end"]
        
        chosen_frames = all_frames[start_chosen:end_chosen + 1]
        rejected_frames = all_frames[start_rejected:end_rejected + 1]
        
        if len(chosen_frames) == 0 or len(rejected_frames) == 0:
            print(f"Skipping sample {i+1}: could not extract frames")
            continue
        
        # Create visualization
        max_frames = max(len(chosen_frames), len(rejected_frames))
        fig, axes = plt.subplots(2, max_frames, figsize=(max_frames * 2, 8))
        
        if max_frames == 1:
            axes = axes.reshape(2, 1)
        
        # Plot chosen frames (top row)
        for j, frame in enumerate(chosen_frames):
            if j < max_frames:
                axes[0, j].imshow(frame)
                axes[0, j].set_title(f"Chosen Frame {j+1}", fontsize=10, color='green', weight='bold')
                axes[0, j].axis('off')
        
        # Fill remaining slots with empty plots
        for j in range(len(chosen_frames), max_frames):
            axes[0, j].text(0.5, 0.5, "N/A", ha='center', va='center', transform=axes[0, j].transAxes)
            axes[0, j].axis('off')
        
        # Plot rejected frames (bottom row)
        for j, frame in enumerate(rejected_frames):
            if j < max_frames:
                axes[1, j].imshow(frame)
                axes[1, j].set_title(f"Rejected Frame {j+1}", fontsize=10, color='red', weight='bold')
                axes[1, j].axis('off')
        
        # Fill remaining slots with empty plots
        for j in range(len(rejected_frames), max_frames):
            axes[1, j].text(0.5, 0.5, "N/A", ha='center', va='center', transform=axes[1, j].transAxes)
            axes[1, j].axis('off')
        
        # Add sample info
        info_text = f"Task: {result.get('chosen_task', 'unknown')}\n"
        info_text += f"Video: {Path(result['video_path']).name}\n"
        info_text += f"Chosen: frames {result['chosen_start_end'][0]}-{result['chosen_start_end'][1]} (bin {result.get('bin_idx_chosen', 'N/A')})\n"
        info_text += f"Rejected: frames {result['rejected_start_end'][0]}-{result['rejected_start_end'][1]} (bin {result.get('bin_idx_rejected', 'N/A')})\n"
        info_text += f"FPS: {result['fps']}\n"
        info_text += f"Preference Label: {result.get('preference_label', 'N/A')}\n"
        info_text += f"Predicted: {result.get('predicted_preference', 'N/A')} (prob: {result.get('predicted_preference_prob', 'N/A'):.3f})"
        
        fig.suptitle(f"Video Binned Sample {i+1}: - Chosen vs Rejected Frame Sequences", 
                    fontsize=14, weight="bold", y=0.95)
        
        # Add info text below the plots
        fig.text(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"video_binned_sample_{i+1}_frames.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Created frame visualization for video binned sample {i+1}")


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a nicely formatted metrics summary."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    print(f"\nPreference Accuracy: {metrics.get('preference_accuracy', 'N/A'):.4f}")

    print(f"\nProgress Metrics (Chosen Trajectory):")
    print(
        f"  MSE: {metrics.get('mse_progress_chosen', 'N/A'):.6f} ± {metrics.get('mse_progress_chosen_std', 'N/A'):.6f}"
    )
    print(
        f"  Spearman Correlation: {metrics.get('spearman_progress_chosen', 'N/A'):.4f} ± {metrics.get('spearman_progress_chosen_std', 'N/A'):.4f}"
    )

    print(f"\nProgress Metrics (Rejected Trajectory):")
    print(
        f"  MSE: {metrics.get('mse_progress_rejected', 'N/A'):.6f} ± {metrics.get('mse_progress_rejected_std', 'N/A'):.6f}"
    )
    print(
        f"  Spearman Correlation: {metrics.get('spearman_progress_rejected', 'N/A'):.4f} ± {metrics.get('spearman_progress_rejected_std', 'N/A'):.4f}"
    )

    print(f"\nData Generation Strategy Counts:")
    for strategy, count in metrics.get("strategy_counts", {}).items():
        print(f"  {strategy}: {count}")

    print(f"\nRewound Frame Counts:")
    for rewound, count in metrics.get("rewound_counts", {}).items():
        print(f"  {rewound} frames: {count}")

    # Bin delta accuracy summary
    if "bin_delta_accuracy" in metrics:
        print(f"\nBin Delta Analysis:")
        print(f"  Overall Accuracy: {metrics.get('bin_delta_overall_accuracy', 'N/A'):.4f}")
        print(f"  Overall Abstention Rate: {metrics.get('bin_delta_overall_abstention_rate', 'N/A'):.4f}")
        print(f"  Total Samples: {metrics.get('bin_delta_total_samples', 'N/A')}")

        delta_accuracy = metrics["bin_delta_accuracy"]
        deltas = sorted(delta_accuracy.keys())
        print(f"  Delta Range: {min(deltas)} to {max(deltas)} bins")

        # Calculate overall abstention rate
        total_abstained = sum(delta_accuracy[delta]["abstained"] for delta in deltas)
        overall_abstention_rate = (
            total_abstained / metrics.get("bin_delta_total_samples", 1)
            if metrics.get("bin_delta_total_samples", 0) > 0
            else 0
        )
        print(f"  Overall Abstention Rate: {overall_abstention_rate:.4f}")

        # Show top 3 most common deltas
        delta_counts = [(delta, delta_accuracy[delta]["total"]) for delta in deltas]
        delta_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"  Most Common Deltas:")
        for i, (delta, count) in enumerate(delta_counts[:3]):
            accuracy = delta_accuracy[delta]["accuracy"]
            abstention_rate = delta_accuracy[delta]["abstention_rate"]
            print(f"    Delta {delta}: {accuracy:.3f} accuracy, {abstention_rate:.3f} abstention ({count} samples)")

    print("\n" + "=" * 80)


def print_bin_delta_summary(delta_accuracy: Dict[str, Any]):
    """Print a summary of bin delta accuracy analysis."""
    if not delta_accuracy:
        print("No bin delta analysis data available")
        return

    print("\n" + "=" * 80)
    print("BIN DELTA ACCURACY ANALYSIS")
    print("=" * 80)

    # Sort deltas for consistent display
    deltas = sorted(delta_accuracy.keys())

    print(f"\nFrame Delta Analysis (|bin_chosen - bin_rejected|):")
    print(f"{'Delta':<8} {'Accuracy':<12} {'Abstained':<12} {'Correct':<10} {'Incorrect':<10} {'Total':<8} {'%':<6}")
    print("-" * 70)

    for delta in deltas:
        data = delta_accuracy[delta]
        accuracy = data["accuracy"]
        abstention_rate = data["abstention_rate"]
        correct = data["correct"]
        incorrect = data["incorrect"]
        abstained = data["abstained"]
        total = data["total"]
        percentage = (correct / total * 100) if total > 0 else 0

        print(
            f"{delta:<8} {accuracy:<12.3f} {abstention_rate:<12.3f} {correct:<10} {incorrect:<10} {total:<8} {percentage:<6.1f}%"
        )

    # Overall statistics
    total_correct = sum(delta_accuracy[delta]["correct"] for delta in delta_accuracy)
    total_incorrect = sum(delta_accuracy[delta]["incorrect"] for delta in delta_accuracy)
    total_abstained = sum(delta_accuracy[delta]["abstained"] for delta in delta_accuracy)
    total_samples = sum(delta_accuracy[delta]["total"] for delta in delta_accuracy)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    overall_abstention_rate = total_abstained / total_samples if total_samples > 0 else 0

    print("-" * 70)
    print(
        f"{'Overall':<8} {overall_accuracy:<12.3f} {overall_abstention_rate:<12.3f} {total_correct:<10} {total_incorrect:<10} {total_samples:<8} {overall_accuracy * 100:<6.1f}%"
    )

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
    parser.add_argument("--results_file", type=str, default=None, help="Path to results.json file")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to visualize")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    results = load_results(str(args.results_file))
    print(f"Loaded {len(results)} samples from {args.results_file}")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # Print metrics summary
    print_metrics_summary(metrics)

    # Analyze bin delta accuracy
    delta_accuracy = analyze_bin_delta_accuracy(results)
    print_bin_delta_summary(delta_accuracy)

    # Create visualizations in the same directory as results.json
    results_dir = Path(args.results_file).parent
    print(f"Creating visualizations in: {results_dir}")
    create_visualizations(results, results_dir, args.max_samples)

    # Create bin delta plot
    create_bin_delta_plot(delta_accuracy, results_dir)

    # Save metrics to file in the same directory as results.json
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    print("Done!")


if __name__ == "__main__":
    main()
