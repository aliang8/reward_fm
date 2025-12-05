#!/usr/bin/env python3
"""
Script to compute detailed statistics for each processed dataset.

This script reads the preprocess.yaml configuration and computes statistics
about frames, trajectories, and other metrics for each dataset/subset combination.
"""

import os
import json
import yaml
import numpy as np
from pathlib import Path
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint
from typing import Dict, List, Any


def load_preprocess_config(config_path: str = "rfm/configs/preprocess.yaml") -> dict:
    """Load the preprocess configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_cache_dir_for_dataset(base_cache_dir: str, dataset_path: str, subset: str) -> str:
    """
    Construct the cache directory path for a dataset/subset pair.

    Following the logic from rfm/data/scripts/preprocess_datasets.py:
    cache_key = f"{dataset_path}/{subset}"
    individual_cache_dir = os.path.join(cache_dir, cache_key.replace("/", "_").replace(":", "_"))
    """
    cache_key = f"{dataset_path}/{subset}"
    individual_cache_dir = os.path.join(base_cache_dir, cache_key.replace("/", "_").replace(":", "_"))
    return individual_cache_dir


def compute_dataset_statistics(cache_dir: str, dataset_path: str, subset: str) -> Dict[str, Any]:
    """
    Compute detailed statistics for a processed dataset.

    Returns a dictionary with:
    - 'num_trajectories': number of trajectories
    - 'total_frames': total number of frames across all trajectories
    - 'frames_per_trajectory': statistics about frames per trajectory (min, max, mean, median, std)
    - 'exists': whether the dataset exists
    - 'error': any error message
    """
    result = {
        "dataset": dataset_path,
        "subset": subset,
        "num_trajectories": 0,
        "total_frames": 0,
        "frames_per_trajectory": {},
        "quality_labels_available": [],
        "exists": False,
        "error": None,
    }

    # Check if cache directory exists
    if not os.path.exists(cache_dir):
        result["error"] = "Cache directory not found"
        return result

    # Check for dataset_info.json
    info_file = os.path.join(cache_dir, "dataset_info.json")
    if not os.path.exists(info_file):
        result["error"] = "dataset_info.json not found"
        return result

    # Check for processed_dataset directory
    dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
    if not os.path.exists(dataset_cache_dir):
        result["error"] = "processed_dataset directory not found"
        return result

    result["exists"] = True

    try:
        # Load the dataset
        dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=False)
        result["num_trajectories"] = len(dataset)

        # Load dataset_info.json for additional metadata
        with open(info_file, "r") as f:
            info = json.load(f)
            result["dataset_info"] = info

        # Load index mappings to get quality labels
        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        if os.path.exists(mappings_file):
            with open(mappings_file, "r") as f:
                indices = json.load(f)
                quality_indices = indices.get("quality_indices", {})
                # Get list of quality labels that have trajectories
                result["quality_labels_available"] = sorted(
                    [label for label, traj_indices in quality_indices.items() if len(traj_indices) > 0]
                )

        # Compute frame statistics
        # Get frames_shape column directly (much faster than iterating)
        if "frames_shape" in dataset.column_names:
            frames_shapes = dataset["frames_shape"]
            # Extract first dimension (number of frames) from each trajectory
            frame_counts = np.array([shape[0] for shape in frames_shapes])
            total_frames = int(np.sum(frame_counts))
        else:
            frame_counts = np.array([])
            total_frames = 0

        result["total_frames"] = total_frames

        if len(frame_counts) > 0:
            result["frames_per_trajectory"] = {
                "min": int(np.min(frame_counts)),
                "max": int(np.max(frame_counts)),
                "mean": float(np.mean(frame_counts)),
                "median": float(np.median(frame_counts)),
                "std": float(np.std(frame_counts)),
                "q25": float(np.percentile(frame_counts, 25)),
                "q75": float(np.percentile(frame_counts, 75)),
            }
        else:
            result["error"] = "Could not determine frame counts from trajectory data"

    except Exception as e:
        result["error"] = f"Error loading dataset: {str(e)}"

    return result


def main():
    """Main function to compute statistics for all processed datasets."""
    console = Console()

    # Get the processed datasets path from environment variable
    cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH")

    if not cache_dir:
        console.print("[bold red]Error:[/bold red] RFM_PROCESSED_DATASETS_PATH environment variable is not set!")
        console.print("Please set it to the directory containing your processed datasets.")
        console.print("Example: export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets")
        return

    console.print(f"[bold]Using cache directory:[/bold] {cache_dir}")

    # Load preprocess configuration
    config_path = "rfm/configs/preprocess.yaml"
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error:[/bold red] Config file not found: {config_path}")
        return

    config = load_preprocess_config(config_path)
    console.print(f"[bold]Loaded config from:[/bold] {config_path}\n")

    # Process training datasets
    train_datasets = config.get("train_datasets", [])
    train_subsets = config.get("train_subsets", [])

    # Process evaluation datasets
    eval_datasets = config.get("eval_datasets", [])
    eval_subsets = config.get("eval_subsets", [])

    # Collect all dataset/subset pairs
    all_datasets = []
    for dataset_path, dataset_subsets in zip(train_datasets, train_subsets):
        for subset in dataset_subsets:
            all_datasets.append(("train", dataset_path, subset))

    for dataset_path, dataset_subsets in zip(eval_datasets, eval_subsets):
        for subset in dataset_subsets:
            all_datasets.append(("eval", dataset_path, subset))

    # Process datasets with progress bar
    train_results = []
    eval_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing dataset statistics...", total=len(all_datasets))

        for split, dataset_path, subset in all_datasets:
            progress.update(task, description=f"[cyan]Processing {dataset_path}/{subset}...")

            individual_cache_dir = get_cache_dir_for_dataset(cache_dir, dataset_path, subset)
            result = compute_dataset_statistics(individual_cache_dir, dataset_path, subset)

            if split == "train":
                train_results.append(result)
            else:
                eval_results.append(result)

            progress.advance(task)

    # Create detailed tables
    console.print("\n")

    # Training datasets table
    train_table = Table(
        title="Training Datasets - Detailed Statistics",
        show_header=True,
        header_style="bold magenta",
    )
    train_table.add_column("Dataset", style="cyan", width=30)
    train_table.add_column("Subset", style="green", width=25)
    train_table.add_column("Trajectories", justify="right", style="yellow")
    train_table.add_column("Total Frames", justify="right", style="yellow")
    train_table.add_column("Frames/Traj\n(mean±std)", justify="right", style="blue")
    train_table.add_column("Min/Max", justify="right", style="white")
    train_table.add_column("Quality Labels", style="magenta", width=30)
    train_table.add_column("Status", style="white")

    train_total_trajectories = 0
    train_total_frames = 0
    train_found = 0

    for result in train_results:
        if result["exists"] and result["error"] is None:
            traj_str = f"{result['num_trajectories']:,}"
            frames_str = f"{result['total_frames']:,}"

            stats = result["frames_per_trajectory"]
            if stats:
                mean_std_str = f"{stats['mean']:.1f}±{stats['std']:.1f}"
                min_max_str = f"{stats['min']}/{stats['max']}"
            else:
                mean_std_str = "N/A"
                min_max_str = "N/A"

            status = "✓"
            status_style = "green"
            train_total_trajectories += result["num_trajectories"]
            train_total_frames += result["total_frames"]
            train_found += 1
            quality_labels_str = ", ".join(result.get("quality_labels_available", [])) or "None"
        else:
            traj_str = "N/A"
            frames_str = "N/A"
            mean_std_str = "N/A"
            min_max_str = "N/A"
            quality_labels_str = "N/A"
            status = f"✗ {result['error']}"
            status_style = "red"

        train_table.add_row(
            result["dataset"],
            result["subset"],
            traj_str,
            frames_str,
            mean_std_str,
            min_max_str,
            quality_labels_str,
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(train_table)
    console.print()

    # Evaluation datasets table
    eval_table = Table(
        title="Evaluation Datasets - Detailed Statistics",
        show_header=True,
        header_style="bold magenta",
    )
    eval_table.add_column("Dataset", style="cyan", width=30)
    eval_table.add_column("Subset", style="green", width=25)
    eval_table.add_column("Trajectories", justify="right", style="yellow")
    eval_table.add_column("Total Frames", justify="right", style="yellow")
    eval_table.add_column("Frames/Traj\n(mean±std)", justify="right", style="blue")
    eval_table.add_column("Min/Max", justify="right", style="white")
    eval_table.add_column("Quality Labels", style="magenta", width=30)
    eval_table.add_column("Status", style="white")

    eval_total_trajectories = 0
    eval_total_frames = 0
    eval_found = 0

    for result in eval_results:
        if result["exists"] and result["error"] is None:
            traj_str = f"{result['num_trajectories']:,}"
            frames_str = f"{result['total_frames']:,}"

            stats = result["frames_per_trajectory"]
            if stats:
                mean_std_str = f"{stats['mean']:.1f}±{stats['std']:.1f}"
                min_max_str = f"{stats['min']}/{stats['max']}"
            else:
                mean_std_str = "N/A"
                min_max_str = "N/A"

            status = "✓"
            status_style = "green"
            eval_total_trajectories += result["num_trajectories"]
            eval_total_frames += result["total_frames"]
            eval_found += 1
            quality_labels_str = ", ".join(result.get("quality_labels_available", [])) or "None"
        else:
            traj_str = "N/A"
            frames_str = "N/A"
            mean_std_str = "N/A"
            min_max_str = "N/A"
            quality_labels_str = "N/A"
            status = f"✗ {result['error']}"
            status_style = "red"

        eval_table.add_row(
            result["dataset"],
            result["subset"],
            traj_str,
            frames_str,
            mean_std_str,
            min_max_str,
            quality_labels_str,
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(eval_table)
    console.print()

    # Summary table
    summary_table = Table(title="Summary Statistics", show_header=True, header_style="bold magenta")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Datasets Found", justify="right", style="green")
    summary_table.add_column("Total Trajectories", justify="right", style="yellow")
    summary_table.add_column("Total Frames", justify="right", style="yellow")
    summary_table.add_column("Avg Frames/Traj", justify="right", style="blue")

    train_avg_frames = train_total_frames / train_total_trajectories if train_total_trajectories > 0 else 0
    eval_avg_frames = eval_total_frames / eval_total_trajectories if eval_total_trajectories > 0 else 0
    total_avg_frames = (
        (train_total_frames + eval_total_frames) / (train_total_trajectories + eval_total_trajectories)
        if (train_total_trajectories + eval_total_trajectories) > 0
        else 0
    )

    summary_table.add_row(
        "Training",
        str(train_found),
        f"{train_total_trajectories:,}",
        f"{train_total_frames:,}",
        f"{train_avg_frames:.1f}",
    )
    summary_table.add_row(
        "Evaluation",
        str(eval_found),
        f"{eval_total_trajectories:,}",
        f"{eval_total_frames:,}",
        f"{eval_avg_frames:.1f}",
    )
    summary_table.add_row(
        "[bold]Grand Total[/bold]",
        f"[bold]{train_found + eval_found}[/bold]",
        f"[bold]{train_total_trajectories + eval_total_trajectories:,}[/bold]",
        f"[bold]{train_total_frames + eval_total_frames:,}[/bold]",
        f"[bold]{total_avg_frames:.1f}[/bold]",
    )

    console.print(summary_table)
    console.print()

    # Save results to JSON
    output_file = "dataset_statistics.json"

    results_json = {
        "cache_dir": cache_dir,
        "training_datasets": train_results,
        "evaluation_datasets": eval_results,
        "summary": {
            "train_datasets_found": train_found,
            "train_total_trajectories": train_total_trajectories,
            "train_total_frames": train_total_frames,
            "train_avg_frames_per_trajectory": train_avg_frames,
            "eval_datasets_found": eval_found,
            "eval_total_trajectories": eval_total_trajectories,
            "eval_total_frames": eval_total_frames,
            "eval_avg_frames_per_trajectory": eval_avg_frames,
            "total_datasets_found": train_found + eval_found,
            "total_trajectories": train_total_trajectories + eval_total_trajectories,
            "total_frames": train_total_frames + eval_total_frames,
            "overall_avg_frames_per_trajectory": total_avg_frames,
        },
    }

    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    console.print(f"[bold green]Results saved to:[/bold green] {output_file}")


if __name__ == "__main__":
    main()
