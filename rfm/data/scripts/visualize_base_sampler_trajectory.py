#!/usr/bin/env python3
"""
Script to visualize trajectories from the base sampler.

Loads a dataset, creates a BaseSampler, processes trajectories using _get_traj_from_data,
and generates videos with overlayed metadata (task, progress, etc.).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from datasets import Dataset
import imageio

# Add parent directory to path to import RFM modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    show_available_datasets,
    load_dataset_success_percent,
)
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.dataset_types import Trajectory
from rfm.utils.distributed import rank_0_print


def load_dataset_from_cache(dataset_name: str) -> tuple[Dataset, str]:
    """Load a dataset from the preprocessed cache.

    Args:
        dataset_name: Name of the dataset (e.g., "metaworld/assembly-v2")

    Returns:
        Tuple of (Dataset, cache_directory_path)
    """
    cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError(
            "RFM_PROCESSED_DATASETS_PATH environment variable not set. "
            "Please set it to the directory containing your processed datasets."
        )

    # The preprocessing script creates individual cache directories for each dataset
    individual_cache_dir = os.path.join(cache_dir, dataset_name.replace("/", "_").replace(":", "_"))

    if not os.path.exists(individual_cache_dir):
        print(f"Error: Cache directory not found: {individual_cache_dir}")
        print("\nAvailable datasets:")
        show_available_datasets()
        raise ValueError(f"Dataset '{dataset_name}' not found in cache")

    info_file = os.path.join(individual_cache_dir, "dataset_info.json")
    if not os.path.exists(info_file):
        raise ValueError(f"Info file not found: {info_file}")

    # Load the processed dataset
    dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
    if not os.path.exists(dataset_cache_dir):
        raise ValueError(f"Processed dataset not found: {dataset_cache_dir}")

    dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=True)
    print(f"✅ Loaded {len(dataset)} trajectories from {dataset_name}")

    return dataset, individual_cache_dir


def load_combined_indices(individual_cache_dir: str) -> dict:
    """Load combined indices from cache directory.

    Args:
        individual_cache_dir: Individual cache directory path for the dataset

    Returns:
        Dictionary of combined indices
    """
    mappings_file = os.path.join(individual_cache_dir, "index_mappings.json")
    if not os.path.exists(mappings_file):
        raise ValueError(f"Index mappings file not found: {mappings_file}")

    with open(mappings_file) as f:
        indices = json.load(f)

    # Initialize combined_indices structure (no offset needed for single dataset)
    combined_indices = {
        "robot_trajectories": indices.get("robot_trajectories", []),
        "human_trajectories": indices.get("human_trajectories", []),
        "optimal_by_task": indices.get("optimal_by_task", {}),
        "suboptimal_by_task": indices.get("suboptimal_by_task", {}),
        "quality_indices": indices.get("quality_indices", {}),
        "task_indices": indices.get("task_indices", {}),
        "source_indices": indices.get("source_indices", {}),
        "partial_success_indices": indices.get("partial_success_indices", {}),
    }

    return combined_indices


def add_text_to_frame(frame: np.ndarray, text_lines: list[str], font_scale: float = 0.4) -> np.ndarray:
    """Add text overlay to a frame.

    Args:
        frame: Input frame (numpy array, H x W x C)
        text_lines: List of text lines to add
        font_scale: Font scale factor

    Returns:
        Frame with text overlay
    """
    frame_copy = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_spacing = 18

    # Starting position (top-left with some padding)
    y_offset = 20
    x_offset = 10

    for i, line in enumerate(text_lines):
        y = y_offset + i * line_spacing

        # Add black background for better text visibility
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(
            frame_copy,
            (x_offset - 3, y - text_height - 3),
            (x_offset + text_width + 3, y + baseline + 3),
            (0, 0, 0),
            -1,
        )

        # Add white text
        cv2.putText(
            frame_copy,
            line,
            (x_offset, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return frame_copy


def create_video_from_trajectory(
    trajectory: Trajectory,
    output_path: str,
    fps: int = 1,
    use_embeddings: bool = False,
) -> None:
    """Create a video from a trajectory with text overlay.

    Args:
        trajectory: Trajectory object
        output_path: Path to save the output video
        fps: Frames per second for the video
        use_embeddings: Whether trajectory uses embeddings (not supported for visualization)
    """
    if use_embeddings:
        raise ValueError("Cannot visualize trajectories with embeddings. Use frames instead.")

    if trajectory.frames is None:
        raise ValueError("Trajectory has no frames to visualize")

    # Load frames if they are paths
    if isinstance(trajectory.frames, str):
        frames = load_frames_from_npz(trajectory.frames)
    elif isinstance(trajectory.frames, np.ndarray):
        frames = trajectory.frames
    elif isinstance(trajectory.frames, list):
        if all(isinstance(f, str) for f in trajectory.frames):
            # List of frame paths - would need to load each, but for now error
            raise ValueError("List of frame paths not supported. Expected npz file path or numpy array.")
        else:
            frames = np.array(trajectory.frames)
    else:
        raise ValueError(f"Unsupported frame format: {type(trajectory.frames)}")

    # Ensure frames are in correct format (T, H, W, C)
    if len(frames.shape) == 4:
        num_frames, height, width, channels = frames.shape
    else:
        raise ValueError(f"Expected frames with shape (T, H, W, C), got {frames.shape}")

    # Get frame dimensions before upscaling
    orig_height, orig_width = height, width
    
    # Upscale frames if they're too small (minimum 512 pixels on shortest side)
    scale_factor = 1.0
    min_dimension = min(height, width)
    if min_dimension < 512:
        scale_factor = 512.0 / min_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # Resize frames
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
        frames = np.array(resized_frames)
        height, width = new_height, new_width
        print(f"Upscaled frames from {orig_width}x{orig_height} to {width}x{height}")

    # Get target progress (default to [0.0] * num_frames if not available)
    if trajectory.target_progress is not None:
        progress_values = trajectory.target_progress[:num_frames]
        # Pad if needed
        while len(progress_values) < num_frames:
            progress_values.append(progress_values[-1] if progress_values else 0.0)
    else:
        progress_values = [0.0] * num_frames

    # Get metadata
    task = trajectory.task or "Unknown"
    metadata = trajectory.metadata or {}
    data_source = trajectory.data_source or "Unknown"
    quality_label = trajectory.quality_label or "Unknown"

    # Format metadata text
    metadata_lines = []
    metadata_lines.append(f"Task: {task}")
    metadata_lines.append(f"Data Source: {data_source}")
    metadata_lines.append(f"Quality: {quality_label}")

    if metadata:
        if "start_idx" in metadata:
            metadata_lines.append(f"Start Idx: {metadata['start_idx']}")
        if "end_idx" in metadata:
            metadata_lines.append(f"End Idx: {metadata['end_idx']}")
        if "subsampled_indices" in metadata:
            metadata_lines.append(f"Indices: {metadata['subsampled_indices'][:5]}...")  # Show first 5
        if "pair_type" in metadata:
            metadata_lines.append(f"Pair Type: {metadata['pair_type']}")
        if "i" in metadata:
            metadata_lines.append(f"i: {metadata['i']}")
        if "T" in metadata:
            metadata_lines.append(f"T: {metadata['T']}")

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Process frames and prepare for video writing
    frame_list = []
    for i, frame in enumerate(frames):
        # Ensure frame is uint8 and in correct range [0, 255]
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert to RGB (imageio uses RGB)
        if channels == 3:
            frame_rgb = frame.copy()
        elif channels == 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

        # Add text overlay
        text_lines = metadata_lines.copy()
        text_lines.append("")  # Empty line for spacing
        text_lines.append(f"Frame: {i+1}/{num_frames}")
        text_lines.append(f"Progress: {progress_values[i]:.3f}")

        frame_with_text = add_text_to_frame(frame_rgb, text_lines)
        frame_list.append(frame_with_text)

    # Write video using imageio (simpler and more reliable)
    imageio.mimwrite(output_path, frame_list, fps=fps)
    print(f"✅ Saved video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectories from base sampler by creating videos with metadata overlay"
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset name (e.g., 'metaworld/assembly-v2' or 'rh20t:rh20t_robot')",
    )
    parser.add_argument(
        "--trajectory_idx",
        type=int,
        default=0,
        help="Index of trajectory to use from the dataset (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_sampler_trajectory",
        help="Output video path or folder name (default: base_sampler_trajectory)",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate (default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for output video (default: 1)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Maximum number of frames (default: 16)",
    )
    parser.add_argument(
        "--progress_pred_type",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help="Progress prediction type (default: absolute)",
    )
    parser.add_argument(
        "--pairwise_progress",
        action="store_true",
        help="Use pairwise progress sampling",
    )
    parser.add_argument(
        "--load_embeddings",
        action="store_true",
        help="Load embeddings instead of frames (not supported for visualization)",
    )
    parser.add_argument(
        "--dataset_success_cutoff_file",
        type=str,
        default=None,
        help="Path to dataset-specific success cutoff file",
    )

    args = parser.parse_args()

    if args.list_datasets:
        show_available_datasets()
        return

    if args.dataset is None:
        parser.error("--dataset is required unless --list_datasets is specified")

    # Create a minimal config object
    class SimpleConfig:
        def __init__(self, args):
            self.max_frames = args.max_frames
            self.progress_pred_type = args.progress_pred_type
            self.pairwise_progress = args.pairwise_progress
            self.load_embeddings = args.load_embeddings
            self.max_success = 0.95  # Default
            self.max_frames_after_preprocessing = 64  # Default, adjust if needed

    config = SimpleConfig(args)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, individual_cache_dir = load_dataset_from_cache(args.dataset)

    # Load combined indices
    combined_indices = load_combined_indices(individual_cache_dir)

    # Load dataset success cutoff map
    dataset_success_cutoff_map = {}
    if args.dataset_success_cutoff_file:
        dataset_success_cutoff_map = load_dataset_success_percent(args.dataset_success_cutoff_file)

    # Create base sampler
    print("Initializing BaseSampler...")
    sampler = RFMBaseSampler(
        config=config,
        dataset=dataset,
        combined_indices=combined_indices,
        dataset_success_cutoff_map=dataset_success_cutoff_map,
        verbose=True,
    )

    # Set up output folder
    if args.num_videos > 1:
        # If multiple videos, create a folder
        output_folder = args.output
        if output_folder.endswith('.mp4'):
            # Remove .mp4 extension if present
            output_folder = os.path.splitext(output_folder)[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created output folder: {output_folder}")
    else:
        # Single video - use output path as-is
        output_folder = None

    # Generate videos
    num_successful = 0
    num_failed = 0
    
    for video_idx in range(args.num_videos):
        print(f"\n{'='*60}")
        print(f"Creating video {video_idx + 1}/{args.num_videos}")
        print(f"{'='*60}")
        
        # Select trajectory (either use specified index or random)
        if args.num_videos == 1 and args.trajectory_idx >= 0:
            traj_idx = args.trajectory_idx
        else:
            # For multiple videos, use random trajectories
            traj_idx = random.randint(0, len(dataset) - 1)
        
        if traj_idx >= len(dataset):
            print(f"Error: Trajectory index {traj_idx} is out of range (dataset has {len(dataset)} trajectories)")
            num_failed += 1
            continue

        # Get trajectory from dataset
        trajectory_dict = dataset[traj_idx]
        print(f"Using trajectory {traj_idx}: {trajectory_dict.get('id', 'unknown')}")

        # Process trajectory using base sampler's _get_traj_from_data
        print("Processing trajectory with BaseSampler...")
        try:
            processed_trajectory = sampler._get_traj_from_data(trajectory_dict)
            print("target: ", processed_trajectory.target_progress[-1])
        except Exception as e:
            print(f"Error: Failed to process trajectory: {e}")
            num_failed += 1
            continue

        print("Trajectory processed successfully!")
        print(f"  Task: {processed_trajectory.task}")
        print(f"  Frames: {len(processed_trajectory.frames) if processed_trajectory.frames is not None else 'N/A'}")
        print(f"  Progress values: {processed_trajectory.target_progress}")

        # Determine output path
        if args.num_videos > 1:
            # Multiple videos: save in folder with indexed name
            output_filename = f"base_sampler_trajectory_{video_idx + 1:04d}.mp4"
            output_path = os.path.join(output_folder, output_filename)
        else:
            # Single video: use specified path or default
            if output_folder and not args.output.endswith('.mp4'):
                output_path = os.path.join(output_folder, "base_sampler_trajectory.mp4")
            elif not args.output.endswith('.mp4'):
                output_path = args.output + ".mp4"
            else:
                output_path = args.output

        # Create video
        print(f"\nCreating video with {args.fps} fps...")
        try:
            create_video_from_trajectory(
                trajectory=processed_trajectory,
                output_path=output_path,
                fps=args.fps,
                use_embeddings=args.load_embeddings,
            )
            num_successful += 1
            print(f"✅ Saved video {video_idx + 1}/{args.num_videos}: {output_path}")
        except Exception as e:
            print(f"❌ Failed to create video {video_idx + 1}: {e}")
            num_failed += 1

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Successful: {num_successful}/{args.num_videos}")
    print(f"  Failed: {num_failed}/{args.num_videos}")
    if args.num_videos > 1 and num_successful > 0:
        print(f"  Output folder: {os.path.abspath(output_folder)}")
    print(f"{'='*60}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

