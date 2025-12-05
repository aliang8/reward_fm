#!/usr/bin/env python3
"""
Script to generate plots of predicted progress values from a JSON dataset.

This script loads the actual dataset using HuggingFace's load_dataset() function
based on the data_source key in the JSON file, fetches trajectories by ID,
and plots the progress values taken from the last item of each progress_pred_A list.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from datasets import load_dataset


def load_progress_data_and_dataset(json_file_path: str) -> tuple[list[dict], dict]:
    """
    Load progress data from JSON file and the corresponding HuggingFace dataset.

    Args:
        json_file_path: Path to the JSON file containing progress data

    Returns:
        Tuple of (progress_data, dataset_dict) where dataset_dict maps data_source to loaded dataset
    """
    with open(json_file_path) as f:
        progress_data = json.load(f)

    # Extract unique data sources
    data_sources = set()
    for item in progress_data:
        data_sources.add(item["data_source"])

    print(f"Found data sources: {list(data_sources)}")

    # Load actual datasets for each data source
    for data_source in data_sources:
        print(f"Loading dataset for data source: {data_source}")

        # Map data_source to actual HuggingFace dataset path
        dataset_path = map_data_source_to_dataset_path(data_source)

        if dataset_path:
            try:
                # Check if we need environment variables
                rfm_dataset_path = os.environ.get("RFM_DATASET_PATH")
                if "/" in dataset_path and not os.path.exists(dataset_path):
                    if not rfm_dataset_path:
                        print("Warning: RFM_DATASET_PATH environment variable not set.")
                        print(f"This may be needed for loading dataset: {dataset_path}")
                        print("Example: export RFM_DATASET_PATH=/path/to/your/datasets")

                dataset = load_dataset(dataset_path, data_source)

                # Patch video paths if needed
                if "/" in dataset_path and rfm_dataset_path:
                    dataset_name = dataset_path.split("/")[-1]

                    def patch_path(old_path):
                        root_dir = f"{rfm_dataset_path}/{dataset_name}"
                        return f"{root_dir}/{old_path}"

                    dataset = dataset.map(
                        lambda x: {"frames_video": patch_path(x["frames"]), "frames_path": patch_path(x["frames"])}
                    )

                print(f"✓ Loaded dataset for {data_source}: {len(dataset)} trajectories")

            except Exception as e:
                print(f"Warning: Failed to load dataset for {data_source}: {e}")
                print("Will proceed with progress data only for this data source.")
        else:
            print(f"Warning: Could not map data_source '{data_source}' to a HuggingFace dataset path")

    return progress_data, dataset


def map_data_source_to_dataset_path(data_source: str) -> str | None:
    """
    Map data_source key to actual HuggingFace dataset path.

    Args:
        data_source: The data_source key from JSON (e.g., "libero256_10")

    Returns:
        HuggingFace dataset path or None if mapping not found
    """
    # Common mappings based on the codebase
    dataset_mappings = {
        "libero256_10": "abraranwar/libero_rfm",
        "libero_rfm": "abraranwar/libero_rfm",
        "libero_failure": "ykorkmaz/libero_failure_rfm",
        "libero_failure_rfm": "ykorkmaz/libero_failure_rfm",
        "metaworld_rewind_eval": "HenryZhang/metaworld_rewind_rfm_eval",
    }

    # Try exact match first
    if data_source in dataset_mappings:
        return dataset_mappings[data_source]

    # Try partial matches for libero datasets
    if "libero" in data_source.lower():
        if "failure" in data_source.lower():
            return "ykorkmaz/libero_failure_rfm"
        else:
            return "abraranwar/libero_rfm"

    # Default fallback - assume it's already a HuggingFace path
    if "/" in data_source:
        return data_source

    return None


def group_trajectories_by_id(data: list[dict]) -> dict[str, list[dict]]:
    """
    Group trajectory data by ID.

    Args:
        data: List of trajectory dictionaries

    Returns:
        Dictionary mapping trajectory IDs to lists of their subsequences
    """
    trajectories = defaultdict(list)

    for item in data:
        trajectory_id = item["id"]
        trajectories[trajectory_id].append(item)

    return dict(trajectories)


def fetch_trajectory_data_from_dataset(progress_data: list[dict], dataset: dict) -> dict[str, dict]:
    """
    Fetch actual trajectory data from the loaded datasets using trajectory IDs.

    Args:
        progress_data: List of progress prediction entries
        dataset_dict: Dictionary mapping data_source to loaded HuggingFace datasets

    Returns:
        Dictionary mapping trajectory IDs to their actual trajectory data
    """
    trajectory_data = {}

    # Create a mapping of trajectory IDs to their data sources
    id_to_data_source = {}
    for item in progress_data:
        trajectory_id = item["id"]
        data_source = item["data_source"]
        id_to_data_source[trajectory_id] = data_source

    # Fetch trajectory data for each unique ID
    unique_ids = set(id_to_data_source.keys())
    print(f"Fetching data for {len(unique_ids)} unique trajectory IDs...")

    for trajectory_id in unique_ids:
        try:
            matching_trajectories = dataset.filter(lambda x: x["id"] == trajectory_id)
            if len(matching_trajectories) > 0:
                trajectory_data[trajectory_id] = matching_trajectories["train"][0]
        except:
            pass

    print(f"✓ Found {len(trajectory_data)} trajectories in datasets")
    return trajectory_data


def calculate_progress_sequences(trajectories: dict[str, list[dict]]) -> dict[str, list[tuple[int, float]]]:
    """
    Calculate progress sequences for each trajectory ID.

    For each trajectory ID:
    1. Sort subsequences by subsequence_end in increasing order
    2. Take the last item from progress_pred_A for each subsequence
    3. Build the progress sequence with indices

    Args:
        trajectories: Dictionary mapping IDs to lists of subsequences

    Returns:
        Dictionary mapping trajectory IDs to their progress sequences with indices
    """
    progress_sequences = {}

    for trajectory_id, subsequences in trajectories.items():
        # Sort subsequences by subsequence_end
        sorted_subsequences = sorted(subsequences, key=lambda x: x["metadata"]["subsequence_end"])
        # Extract the last progress value from each subsequence and include the index
        progress_sequence = []
        for subseq in sorted_subsequences:
            last_progress = subseq["progress_pred_A"][-1]  # Take last item
            index = subseq["metadata"]["subsequence_end"]
            progress_sequence.append((index, last_progress))

        progress_sequences[trajectory_id] = progress_sequence

    return progress_sequences


def plot_progress_sequences(
    progress_sequences: dict[str, list[tuple[int, float]]],
    trajectory_data: dict[str, dict] | None = None,
    output_path: str | None = None,
    show_plot: bool = True,
):
    """
    Generate plots of progress sequences and output frame locations.

    Args:
        progress_sequences: Dictionary mapping trajectory IDs to progress sequences
        trajectory_data: Optional dictionary mapping trajectory IDs to their actual data
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot each trajectory
    for trajectory_id, progress_seq in progress_sequences.items():
        # Add (0, 0) as the initial value
        progress_seq = [(0, 0), *progress_seq]
        subsequence_indices, progress_values = zip(*progress_seq, strict=False)
        plt.plot(
            subsequence_indices,
            progress_values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"Trajectory {trajectory_id[:8]}...",
        )  # Show first 8 chars of ID

        # Output frame locations if available
        if trajectory_data and trajectory_id in trajectory_data:
            traj_info = trajectory_data[trajectory_id]
            if "frames_video" in traj_info:
                print(f"Trajectory {trajectory_id} video location: {traj_info['frames_video']}")

    plt.xlabel("Subsequence Index", fontsize=12)
    plt.ylabel("Predicted Progress", fontsize=12)
    plt.title("Predicted Progress Values by Trajectory", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set y-axis limits to show full range
    plt.ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()


def plot_individual_trajectories(
    progress_sequences: dict[str, list[tuple[int, float]]],
    trajectory_data: dict[str, dict] | None = None,
    output_dir: str | None = None,
):
    """
    Generate individual plots for each trajectory and output frame locations.

    Args:
        progress_sequences: Dictionary mapping trajectory IDs to progress sequences
        trajectory_data: Optional dictionary mapping trajectory IDs to their actual data
        output_dir: Directory to save individual plots
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for trajectory_id, progress_seq in progress_sequences.items():
        plt.figure(figsize=(10, 6))

        # Add (0, 0) as the initial value
        progress_seq = [(0, 0), *progress_seq]
        subsequence_indices, progress_values = zip(*progress_seq, strict=False)
        plt.plot(subsequence_indices, progress_values, marker="o", linewidth=2, markersize=8, color="blue")

        # Output frame locations if available
        if trajectory_data and trajectory_id in trajectory_data:
            traj_info = trajectory_data[trajectory_id]
            if "frames_video" in traj_info:
                print(f"Trajectory {trajectory_id} video location: {traj_info['frames_video']}")

        plt.xlabel("Subsequence Index", fontsize=12)
        plt.ylabel("Predicted Progress", fontsize=12)
        plt.title(f"Predicted Progress - Video: {traj_info['frames']}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        if output_dir:
            output_path = Path(output_dir) / f"trajectory_{trajectory_id[:8]}.png"
            print(f"Saving plot to: {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.show()


def print_summary_statistics(
    progress_sequences: dict[str, list[tuple[int, float]]], trajectory_data: dict[str, dict] | None = None
):
    """
    Print summary statistics of the progress sequences and output frame locations.

    Args:
        progress_sequences: Dictionary mapping trajectory IDs to progress sequences
        trajectory_data: Optional dictionary mapping trajectory IDs to their actual data
    """
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    total_trajectories = len(progress_sequences)
    print(f"Total trajectories: {total_trajectories}")

    if trajectory_data:
        print(f"Trajectories with dataset info: {len(trajectory_data)}")

    for trajectory_id, progress_seq in progress_sequences.items():
        _indices, progress_values = zip(*progress_seq, strict=False)
        print(f"\nTrajectory {trajectory_id}:")
        print(f"  Number of subsequences: {len(progress_values)}")
        print(f"  Progress range: {min(progress_values):.3f} - {max(progress_values):.3f}")
        print(f"  Final progress: {progress_values[-1]:.3f}")
        print(f"  Progress sequence: {[f'{p:.3f}' for p in progress_values]}")

        # Add trajectory info if available
        if trajectory_data and trajectory_id in trajectory_data:
            traj_info = trajectory_data[trajectory_id]
            print("  Dataset info available: ✓")
            if "task" in traj_info:
                print(f"  Task: {traj_info['task']}")
            if "frames_video" in traj_info:
                print("  Has frames: ✓")
                print(f"  Video location: {traj_info['frames_video']}")
            if "actions" in traj_info:
                print("  Has actions: ✓")
        else:
            print("  Dataset info available: ✗")


def main():
    """Main function to run the progress plotting script."""
    parser = argparse.ArgumentParser(description="Plot predicted progress values from reward alignment results")
    parser.add_argument(
        "--json_file",
        default="eval_logs/aliangdw_rfm_prefprog_v4/ykorkmaz_libero_failure_rfm_libero_10_failure/reward_alignment_progress.json",
        help="Path to the JSON file containing reward alignment results",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for saving trajectory plots")

    args = parser.parse_args()

    # Load the progress data and corresponding datasets
    print(f"Loading data from: {args.json_file}")
    # Load both progress data and actual datasets
    progress_data, dataset = load_progress_data_and_dataset(args.json_file)
    print(f"Loaded {len(progress_data)} progress data points")

    # Fetch trajectory data from the loaded datasets
    trajectory_data = fetch_trajectory_data_from_dataset(progress_data, dataset)

    # Group by trajectory ID
    trajectories = group_trajectories_by_id(progress_data)
    print(f"Found {len(trajectories)} unique trajectories")

    # Calculate progress sequences
    progress_sequences = calculate_progress_sequences(trajectories)

    # Print summary
    print_summary_statistics(progress_sequences, trajectory_data)

    # Generate plots
    plot_individual_trajectories(progress_sequences, trajectory_data, args.output_dir)


if __name__ == "__main__":
    main()
