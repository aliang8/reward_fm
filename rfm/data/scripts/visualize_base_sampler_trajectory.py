#!/usr/bin/env python3
"""
Script to visualize trajectories from the base sampler.

Loads a dataset, creates a BaseSampler, processes trajectories using _get_traj_from_data,
and generates videos with overlayed metadata (task, progress, etc.).

Example commands:
    # Visualize multiple preference samples
    uv run python rfm/data/scripts/visualize_base_sampler_trajectory.py \
        --dataset jesbu1/oxe_rfm_oxe_bc_z \
        --viz_type preference \
        --strategy rewound \
        --num_videos 10 \
        --max_frames 2 \
        --progress_pred_type absolute_wrt_total_frames \
        --output data_strategy_visualization

    uv run python rfm/data/scripts/visualize_base_sampler_trajectory.py \
        --dataset jesbu1/oxe_rfm_oxe_bc_z \
        --viz_type progress \
        --strategy rewound \
        --num_videos 10 \
        --max_frames 2 \
        --progress_pred_type absolute_wrt_total_frames \
        --output data_strategy_visualization

    # List available datasets
    uv run python rfm/data/scripts/visualize_base_sampler_trajectory.py --list_datasets
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
    DataGenStrat,
)
from rfm.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.data.dataset_types import Trajectory, PreferenceSample
from rfm.data.datasets.base import BaseDataset
from rfm.configs.experiment_configs import DataConfig
from rfm.utils.distributed import rank_0_print
from rfm.utils.logger import setup_loguru_logging


def get_dataset_short_name(dataset_name: str) -> str:
    """Get the short name for a dataset.

    Args:
        dataset_name: Full dataset name (e.g., "metaworld/assembly-v2")

    Returns:
        Short name from mapping, or a sanitized version of the dataset name
    """
    # Try to get from mapping first
    short_name = DS_SHORT_NAME_MAPPING.get(dataset_name, None)
    if short_name:
        return short_name

    # If not in mapping, create a short name from the dataset name
    # Replace "/" and ":" with "_" and take the last part if it contains "/"
    sanitized = dataset_name.replace("/", "_").replace(":", "_")
    # If it's too long, take a reasonable portion
    if len(sanitized) > 30:
        # Try to get meaningful parts
        parts = dataset_name.replace(":", "/").split("/")
        if len(parts) > 1:
            sanitized = "_".join(parts[-2:])  # Take last two parts
        else:
            sanitized = sanitized[:30]

    return sanitized


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
    partial_success = trajectory.partial_success

    # Format metadata text for header (static info)
    header_lines = []
    header_lines.append(f"Task: {task}")
    header_lines.append(f"Data Source: {data_source}")
    header_lines.append(f"Quality: {quality_label}")
    if partial_success is not None:
        header_lines.append(f"Partial Success: {partial_success:.3f}")

    if metadata:
        if "start_idx" in metadata:
            header_lines.append(f"Start Idx: {metadata['start_idx']}")
        if "end_idx" in metadata:
            header_lines.append(f"End Idx: {metadata['end_idx']}")
        if "subsampled_indices" in metadata:
            header_lines.append(f"Indices: {metadata['subsampled_indices']} ")  # Show first 5
        if "pair_type" in metadata:
            header_lines.append(f"Pair Type: {metadata['pair_type']}")
        if "i" in metadata:
            header_lines.append(f"i: {metadata['i']}")
        if "T" in metadata:
            header_lines.append(f"T: {metadata['T']}")

    # Calculate header height needed for metadata
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_spacing = 20  # Increased spacing to prevent overlap
    # Calculate height: static metadata + spacing + frame info (2 lines) + padding
    static_metadata_height = len(header_lines) * line_spacing
    frame_info_height = 2 * line_spacing  # Frame number + progress
    header_height = static_metadata_height + frame_info_height + 60  # Extra padding

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

        # Create header with metadata
        header = np.zeros((header_height, width, 3), dtype=np.uint8)
        header_with_text = add_text_to_frame(header, header_lines, font_scale=font_scale)

        # Add frame-specific info to header (bottom of header, with proper spacing)
        frame_info_lines = [f"Frame: {i + 1}/{num_frames}", f"Progress: {progress_values[i]:.3f}"]
        # Add frame info starting after static metadata with spacing
        y_offset = static_metadata_height + 30  # Start after static metadata with spacing
        for j, line in enumerate(frame_info_lines):
            y = y_offset + j * line_spacing
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(
                header_with_text,
                (10 - 3, y - text_height - 3),
                (10 + text_width + 3, y + baseline + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                header_with_text,
                line,
                (10, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Concatenate header above the video frame
        frame_with_header = np.concatenate([header_with_text, frame_rgb], axis=0)
        frame_list.append(frame_with_header)

    # Write video using imageio (simpler and more reliable)
    imageio.mimwrite(output_path, frame_list, fps=fps)
    print(f"✅ Saved video to: {output_path}")


def create_preference_video(
    preference_sample: PreferenceSample,
    output_path: str,
    fps: int = 1,
    use_embeddings: bool = False,
) -> None:
    """Create a side-by-side video from a preference sample (chosen vs rejected).

    Args:
        preference_sample: PreferenceSample object with chosen and rejected trajectories
        output_path: Path to save the output video
        fps: Frames per second for the video
        use_embeddings: Whether trajectories use embeddings (not supported for visualization)
    """
    if use_embeddings:
        raise ValueError("Cannot visualize trajectories with embeddings. Use frames instead.")

    chosen_traj = preference_sample.chosen_trajectory
    rejected_traj = preference_sample.rejected_trajectory

    # Load frames for both trajectories
    def load_traj_frames(traj: Trajectory) -> np.ndarray:
        if traj.frames is None:
            raise ValueError("Trajectory has no frames to visualize")
        if isinstance(traj.frames, str):
            return load_frames_from_npz(traj.frames)
        elif isinstance(traj.frames, np.ndarray):
            return traj.frames
        elif isinstance(traj.frames, list):
            if all(isinstance(f, str) for f in traj.frames):
                raise ValueError("List of frame paths not supported. Expected npz file path or numpy array.")
            else:
                return np.array(traj.frames)
        else:
            raise ValueError(f"Unsupported frame format: {type(traj.frames)}")

    chosen_frames = load_traj_frames(chosen_traj)
    rejected_frames = load_traj_frames(rejected_traj)

    # Ensure frames are in correct format (T, H, W, C)
    if len(chosen_frames.shape) != 4 or len(rejected_frames.shape) != 4:
        raise ValueError(
            f"Expected frames with shape (T, H, W, C), got chosen: {chosen_frames.shape}, rejected: {rejected_frames.shape}"
        )

    # Get frame dimensions
    chosen_num_frames, chosen_height, chosen_width, chosen_channels = chosen_frames.shape
    rejected_num_frames, rejected_height, rejected_width, rejected_channels = rejected_frames.shape

    # Upscale frames if needed (minimum 256 pixels on shortest side for side-by-side)
    scale_factor = 1.0
    min_dimension = min(chosen_height, chosen_width, rejected_height, rejected_width)
    if min_dimension < 256:
        scale_factor = 256.0 / min_dimension
        new_chosen_width = int(chosen_width * scale_factor)
        new_chosen_height = int(chosen_height * scale_factor)
        new_rejected_width = int(rejected_width * scale_factor)
        new_rejected_height = int(rejected_height * scale_factor)

        resized_chosen = []
        for frame in chosen_frames:
            resized_frame = cv2.resize(frame, (new_chosen_width, new_chosen_height), interpolation=cv2.INTER_LINEAR)
            resized_chosen.append(resized_frame)
        chosen_frames = np.array(resized_chosen)

        resized_rejected = []
        for frame in rejected_frames:
            resized_frame = cv2.resize(frame, (new_rejected_width, new_rejected_height), interpolation=cv2.INTER_LINEAR)
            resized_rejected.append(resized_frame)
        rejected_frames = np.array(resized_rejected)

        chosen_height, chosen_width = new_chosen_height, new_chosen_width
        rejected_height, rejected_width = new_rejected_height, new_rejected_width
        print(
            f"Upscaled frames to {chosen_width}x{chosen_height} (chosen) and {rejected_width}x{rejected_height} (rejected)"
        )

    # Pad to same number of frames
    max_frames = max(chosen_num_frames, rejected_num_frames)
    if chosen_num_frames < max_frames:
        # Pad chosen frames with last frame
        last_frame = chosen_frames[-1]
        padding = np.repeat(last_frame[np.newaxis, :, :, :], max_frames - chosen_num_frames, axis=0)
        chosen_frames = np.concatenate([chosen_frames, padding], axis=0)
    if rejected_num_frames < max_frames:
        # Pad rejected frames with last frame
        last_frame = rejected_frames[-1]
        padding = np.repeat(last_frame[np.newaxis, :, :, :], max_frames - rejected_num_frames, axis=0)
        rejected_frames = np.concatenate([rejected_frames, padding], axis=0)

    # Resize to same height (use the larger height)
    target_height = max(chosen_height, rejected_height)
    if chosen_height != target_height:
        resized_chosen = []
        for frame in chosen_frames:
            resized_frame = cv2.resize(frame, (chosen_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_chosen.append(resized_frame)
        chosen_frames = np.array(resized_chosen)
        chosen_height = target_height
    if rejected_height != target_height:
        resized_rejected = []
        for frame in rejected_frames:
            resized_frame = cv2.resize(frame, (rejected_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_rejected.append(resized_frame)
        rejected_frames = np.array(resized_rejected)
        rejected_height = target_height

    # Get progress values
    chosen_progress = chosen_traj.target_progress[:max_frames] if chosen_traj.target_progress else [0.0] * max_frames
    rejected_progress = (
        rejected_traj.target_progress[:max_frames] if rejected_traj.target_progress else [0.0] * max_frames
    )
    while len(chosen_progress) < max_frames:
        chosen_progress.append(chosen_progress[-1] if chosen_progress else 0.0)
    while len(rejected_progress) < max_frames:
        rejected_progress.append(rejected_progress[-1] if rejected_progress else 0.0)

    # Get metadata
    task = chosen_traj.task or "Unknown"
    strategy = preference_sample.data_gen_strategy or "Unknown"
    chosen_id = chosen_traj.id or "Unknown"
    rejected_id = rejected_traj.id or "Unknown"
    chosen_partial_success = chosen_traj.partial_success
    rejected_partial_success = rejected_traj.partial_success
    chosen_quality_label = chosen_traj.quality_label or "Unknown"
    rejected_quality_label = rejected_traj.quality_label or "Unknown"

    # Calculate header height needed for metadata
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_spacing = 20  # Increased spacing to prevent overlap

    # Header will contain: strategy, task, chosen/rejected info, and frame-specific info
    header_lines = [
        f"Strategy: {strategy}",
        f"Task: {task}",
        f"Chosen ID: {chosen_id} | Quality: {chosen_quality_label}",
        f"Rejected ID: {rejected_id} | Quality: {rejected_quality_label}",
    ]
    if chosen_partial_success is not None or rejected_partial_success is not None:
        partial_info = []
        if chosen_partial_success is not None:
            partial_info.append(f"Chosen Partial: {chosen_partial_success:.3f}")
        if rejected_partial_success is not None:
            partial_info.append(f"Rejected Partial: {rejected_partial_success:.3f}")
        header_lines.append(" | ".join(partial_info))

    # Calculate height: static metadata + spacing + frame info (2 lines) + padding
    static_metadata_height = len(header_lines) * line_spacing
    frame_info_height = 2 * line_spacing  # Frame number + progress
    header_height = static_metadata_height + frame_info_height + 60  # Extra padding
    header_width = chosen_width + rejected_width  # Total width of side-by-side frames

    # Create side-by-side frames
    frame_list = []
    for i in range(max_frames):
        # Ensure frames are uint8
        chosen_frame = chosen_frames[i]
        rejected_frame = rejected_frames[i]

        if chosen_frame.dtype != np.uint8:
            if chosen_frame.max() <= 1.0:
                chosen_frame = (chosen_frame * 255).astype(np.uint8)
            else:
                chosen_frame = np.clip(chosen_frame, 0, 255).astype(np.uint8)

        if rejected_frame.dtype != np.uint8:
            if rejected_frame.max() <= 1.0:
                rejected_frame = (rejected_frame * 255).astype(np.uint8)
            else:
                rejected_frame = np.clip(rejected_frame, 0, 255).astype(np.uint8)

        # Convert to RGB if needed
        if chosen_channels == 1:
            chosen_frame = cv2.cvtColor(chosen_frame, cv2.COLOR_GRAY2RGB)
        if rejected_channels == 1:
            rejected_frame = cv2.cvtColor(rejected_frame, cv2.COLOR_GRAY2RGB)

        # Create header with metadata
        header = np.zeros((header_height, header_width, 3), dtype=np.uint8)
        header_with_text = add_text_to_frame(header, header_lines, font_scale=font_scale)

        # Add frame-specific info to header (after static metadata with proper spacing)
        frame_info_lines = [
            f"Frame: {i + 1}/{max_frames}",
            f"Chosen Progress: {chosen_progress[i]:.3f} | Rejected Progress: {rejected_progress[i]:.3f}",
        ]
        # Add frame info starting after static metadata with spacing
        y_offset = static_metadata_height + 30  # Start after static metadata with spacing
        for j, line in enumerate(frame_info_lines):
            y = y_offset + j * line_spacing
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(
                header_with_text,
                (10 - 3, y - text_height - 3),
                (10 + text_width + 3, y + baseline + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                header_with_text,
                line,
                (10, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Add visual indicators to frames (CHOSEN on left, REJECTED on right)
        # Add a colored border and label to chosen frame (left side)
        chosen_frame_with_indicator = chosen_frame.copy()
        # Add green border (thickness 5) to indicate chosen
        cv2.rectangle(chosen_frame_with_indicator, (0, 0), (chosen_width - 1, chosen_height - 1), (0, 255, 0), 5)
        # Add "CHOSEN" label at top-left
        cv2.putText(
            chosen_frame_with_indicator,
            "CHOSEN",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Add a colored border and label to rejected frame (right side)
        rejected_frame_with_indicator = rejected_frame.copy()
        # Add red border (thickness 5) to indicate rejected
        cv2.rectangle(rejected_frame_with_indicator, (0, 0), (rejected_width - 1, rejected_height - 1), (0, 0, 255), 5)
        # Add "REJECTED" label at top-left
        cv2.putText(
            rejected_frame_with_indicator,
            "REJECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Concatenate side by side with visual indicators
        side_by_side = np.concatenate([chosen_frame_with_indicator, rejected_frame_with_indicator], axis=1)

        # Concatenate header above the video
        frame_with_header = np.concatenate([header_with_text, side_by_side], axis=0)
        frame_list.append(frame_with_header)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write video
    imageio.mimwrite(output_path, frame_list, fps=fps)
    print(f"✅ Saved preference video to: {output_path}")


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
        default="data_strategy_visualization",
        help="Output video path or folder name (default: data_strategy_visualization)",
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
        default="absolute_first_frame",
        choices=["absolute_first_frame", "relative_first_frame", "absolute_wrt_total_frames"],
        help="Progress prediction type: 'absolute_first_frame', 'relative_first_frame', or 'absolute_wrt_total_frames' (default: absolute_first_frame)",
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
    parser.add_argument(
        "--viz_type",
        type=str,
        default="progress",
        choices=["progress", "preference"],
        help="Visualization type: 'progress' for single progress trajectory or 'preference' for preference sample (default: progress)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=[
            "rewound",
            "suboptimal",
            "different_task",
            "roboarena_partial_success",
            "reverse_progress",
            "successful",
            "subsequence",
            "different_task_instruction",
        ],
        help="Data generation strategy. For preference: rewound, suboptimal, different_task, roboarena_partial_success. For progress: successful, rewound, subsequence, reverse_progress, different_task_instruction.",
    )
    parser.add_argument(
        "--dataset_preference_ratio",
        type=float,
        default=0.0,
        help="Ratio for using dataset preferences (default: 0.0)",
    )
    parser.add_argument(
        "--preference_strategy_ratio",
        type=str,
        default="1,1,1",
        help="Comma-separated ratios for preference strategies [rewound, suboptimal, different_task] (default: 1,1,1)",
    )

    args = parser.parse_args()

    setup_loguru_logging(log_level="TRACE")

    if args.list_datasets:
        show_available_datasets()
        return

    if args.dataset is None:
        parser.error("--dataset is required unless --list_datasets is specified")

    # Parse preference_strategy_ratio
    strategy_ratios = [float(x.strip()) for x in args.preference_strategy_ratio.split(",")]
    if len(strategy_ratios) != 3:
        raise ValueError(f"preference_strategy_ratio must have 3 values, got {len(strategy_ratios)}")

    # Create DataConfig using setup utilities
    data_config = DataConfig(
        train_datasets=[args.dataset],  # Use the specified dataset for both train and eval
        eval_datasets=[args.dataset],
        max_frames=args.max_frames,
        progress_pred_type=args.progress_pred_type,
        pairwise_progress=args.pairwise_progress,
        load_embeddings=args.load_embeddings,
        max_success=0.95,  # Default
        max_frames_after_preprocessing=64,  # Default
        dataset_preference_ratio=args.dataset_preference_ratio,
        preference_strategy_ratio=strategy_ratios,
        dataset_success_cutoff_file=args.dataset_success_cutoff_file,
        min_frames_per_trajectory=0,
        sample_type_ratio=[1, 0, 0] if args.viz_type == "preference" else [0, 1, 0],  # Preference or progress
    )

    # Load dataset using BaseDataset (reuses all the loading logic)
    print(f"Loading dataset: {args.dataset}")
    base_dataset = BaseDataset(config=data_config, is_evaluation=False)

    # Extract dataset and combined_indices from BaseDataset
    dataset = base_dataset.dataset
    combined_indices = base_dataset._combined_indices
    dataset_success_cutoff_map = base_dataset.dataset_success_cutoff_map

    print(f"✅ Loaded {len(dataset)} trajectories from {args.dataset}")

    # Get dataset short name for filename
    ds_short_name = get_dataset_short_name(args.dataset)

    # Create sampler based on visualization type
    if args.viz_type == "preference":
        print("Initializing PrefSampler...")
        sampler = PrefSampler(
            config=data_config,
            dataset=dataset,
            combined_indices=combined_indices,
            dataset_success_cutoff_map=dataset_success_cutoff_map,
            verbose=True,
        )
    else:  # progress
        print("Initializing ProgressSampler...")
        sampler = ProgressSampler(
            config=data_config,
            dataset=dataset,
            combined_indices=combined_indices,
            dataset_success_cutoff_map=dataset_success_cutoff_map,
            verbose=True,
        )

    # Set up output folder
    if args.num_videos > 1:
        # If multiple videos, create a folder
        output_folder = args.output
        if output_folder.endswith(".mp4"):
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
        print(f"\n{'=' * 60}")
        print(f"Creating video {video_idx + 1}/{args.num_videos}")
        print(f"{'=' * 60}")

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

        if args.viz_type == "preference":
            # Generate preference sample
            print("Generating preference sample...")
            try:
                # If strategy is specified, override the strategy ratios
                if args.strategy:
                    strategy_map = {
                        "rewound": DataGenStrat.REWOUND,
                        "suboptimal": DataGenStrat.SUBOPTIMAL,
                        "different_task": DataGenStrat.DIFFERENT_TASK,
                        "roboarena_partial_success": DataGenStrat.ROBOARENA_PARTIAL_SUCCESS,
                    }
                    target_strategy = strategy_map.get(args.strategy)

                    if target_strategy is not None:
                        # Temporarily override strategy ratios to force the desired strategy
                        original_ratios = data_config.preference_strategy_ratio.copy()
                        if target_strategy == DataGenStrat.REWOUND:
                            data_config.preference_strategy_ratio = [1.0, 0.0, 0.0]
                        elif target_strategy == DataGenStrat.SUBOPTIMAL:
                            data_config.preference_strategy_ratio = [0.0, 1.0, 0.0]
                        elif target_strategy == DataGenStrat.DIFFERENT_TASK:
                            data_config.preference_strategy_ratio = [0.0, 0.0, 1.0]
                        elif target_strategy == DataGenStrat.ROBOARENA_PARTIAL_SUCCESS:
                            # For RoboArena, the strategy will be automatically selected if it's RoboArena
                            pass
                        if target_strategy is not None:
                            sampler.preference_strategy_ratio = data_config.preference_strategy_ratio

                # Generate preference sample using _generate_sample (which internally calls _create_pref_sample)
                preference_sample = sampler._generate_sample(trajectory_dict)

                if preference_sample is None:
                    print(f"Error: Failed to generate preference sample for trajectory {traj_idx}")
                    num_failed += 1
                    continue

                print("Preference sample generated successfully!")
                print(f"  Task: {preference_sample.chosen_trajectory.task}")
                print(f"  Strategy: {preference_sample.data_gen_strategy}")
                print(f"  Chosen ID: {preference_sample.chosen_trajectory.id}")
                print(f"  Rejected ID: {preference_sample.rejected_trajectory.id}")
                print(
                    f"  Chosen progress: {preference_sample.chosen_trajectory.target_progress[-1] if preference_sample.chosen_trajectory.target_progress else 'N/A'}"
                )
                print(
                    f"  Rejected progress: {preference_sample.rejected_trajectory.target_progress[-1] if preference_sample.rejected_trajectory.target_progress else 'N/A'}"
                )

                # Print frame indices for both trajectories
                chosen_metadata = preference_sample.chosen_trajectory.metadata or {}
                rejected_metadata = preference_sample.rejected_trajectory.metadata or {}

                if "subsampled_indices" in chosen_metadata:
                    chosen_indices = chosen_metadata["subsampled_indices"]
                    print(f"  Chosen frame indices: {chosen_indices}")
                if "start_idx" in chosen_metadata and "end_idx" in chosen_metadata:
                    print(f"  Chosen frame range: {chosen_metadata['start_idx']} to {chosen_metadata['end_idx']}")
                else:
                    print(f"  Chosen frame indices: All frames (no subsampling)")

                if "subsampled_indices" in rejected_metadata:
                    rejected_indices = rejected_metadata["subsampled_indices"]
                    print(f"  Rejected frame indices: {rejected_indices}")
                if "start_idx" in rejected_metadata and "end_idx" in rejected_metadata:
                    print(f"  Rejected frame range: {rejected_metadata['start_idx']} to {rejected_metadata['end_idx']}")
                else:
                    print(f"  Rejected frame indices: All frames (no subsampling)")

                # Restore original ratios if we overrode them
                if args.strategy:
                    data_config.preference_strategy_ratio = original_ratios
                    sampler.preference_strategy_ratio = original_ratios

            except Exception as e:
                print(f"Error: Failed to generate preference sample: {e}")
                import traceback

                traceback.print_exc()
                num_failed += 1
                continue

            # Determine output path
            strategy_suffix = f"_{args.strategy}" if args.strategy else ""
            if args.num_videos > 1:
                output_filename = f"{ds_short_name}_preference{strategy_suffix}_{video_idx + 1:04d}.mp4"
                output_path = os.path.join(output_folder, output_filename)
            else:
                if output_folder and not args.output.endswith(".mp4"):
                    output_path = os.path.join(output_folder, f"{ds_short_name}_preference{strategy_suffix}.mp4")
                elif not args.output.endswith(".mp4"):
                    output_path = f"{ds_short_name}_preference{strategy_suffix}_{args.output}.mp4"
                else:
                    output_dir = os.path.dirname(args.output)
                    base_filename = os.path.basename(args.output)
                    base_name = os.path.splitext(base_filename)[0]
                    if output_dir:
                        output_path = os.path.join(
                            output_dir, f"{ds_short_name}_preference{strategy_suffix}_{base_name}.mp4"
                        )
                    else:
                        output_path = f"{ds_short_name}_preference{strategy_suffix}_{base_name}.mp4"

            # Create preference video
            print(f"\nCreating preference video with {args.fps} fps...")
            try:
                create_preference_video(
                    preference_sample=preference_sample,
                    output_path=output_path,
                    fps=args.fps,
                    use_embeddings=args.load_embeddings,
                )
                num_successful += 1
                print(f"✅ Saved preference video {video_idx + 1}/{args.num_videos}: {output_path}")
            except Exception as e:
                print(f"❌ Failed to create preference video {video_idx + 1}: {e}")
                import traceback

                traceback.print_exc()
                num_failed += 1
        else:  # progress
            # Generate progress sample
            print("Generating progress sample...")
            try:
                # If strategy is specified, override the strategy ratios
                if args.strategy:
                    strategy_map = {
                        "successful": DataGenStrat.SUCCESSFUL,
                        "rewound": DataGenStrat.REWOUND,
                        "subsequence": DataGenStrat.SUBSEQUENCE,
                        "reverse_progress": DataGenStrat.REVERSE_PROGRESS,
                        "different_task_instruction": DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
                    }
                    target_strategy = strategy_map.get(args.strategy)

                    if target_strategy is not None:
                        # Temporarily override strategy ratios to force the desired strategy
                        original_ratios = data_config.progress_strategy_ratio.copy()
                        # progress_strategy_ratio: [successful, rewind, different_task, subsequence, reverse_progress]
                        if target_strategy == DataGenStrat.SUCCESSFUL:
                            data_config.progress_strategy_ratio = [1.0, 0.0, 0.0, 0.0, 0.0]
                        elif target_strategy == DataGenStrat.REWOUND:
                            data_config.progress_strategy_ratio = [0.0, 1.0, 0.0, 0.0, 0.0]
                        elif target_strategy == DataGenStrat.DIFFERENT_TASK_INSTRUCTION:
                            data_config.progress_strategy_ratio = [0.0, 0.0, 1.0, 0.0, 0.0]
                        elif target_strategy == DataGenStrat.SUBSEQUENCE:
                            data_config.progress_strategy_ratio = [0.0, 0.0, 0.0, 1.0, 0.0]
                        elif target_strategy == DataGenStrat.REVERSE_PROGRESS:
                            data_config.progress_strategy_ratio = [0.0, 0.0, 0.0, 0.0, 1.0]

                # Generate progress sample using _generate_sample (which internally calls _create_progress_sample)
                progress_sample = sampler._generate_sample(trajectory_dict)

                if progress_sample is None:
                    print(f"Error: Failed to generate progress sample for trajectory {traj_idx}")
                    num_failed += 1
                    continue

                processed_trajectory = progress_sample.trajectory

                print("Progress sample generated successfully!")
                print(f"  Task: {processed_trajectory.task}")
                print(f"  Strategy: {progress_sample.data_gen_strategy}")
                print(
                    f"  Frames: {len(processed_trajectory.frames) if processed_trajectory.frames is not None else 'N/A'}"
                )
                print(f"  Progress values: {processed_trajectory.target_progress}")
                print(
                    f"  Final progress: {processed_trajectory.target_progress[-1] if processed_trajectory.target_progress else 'N/A'}"
                )

                # Restore original ratios if we overrode them
                if args.strategy and target_strategy is not None:
                    data_config.progress_strategy_ratio = original_ratios

            except Exception as e:
                print(f"Error: Failed to generate progress sample: {e}")
                import traceback

                traceback.print_exc()
                num_failed += 1
                continue

            # Determine output path
            strategy_suffix = f"_{args.strategy}" if args.strategy else ""
            if args.num_videos > 1:
                # Multiple videos: save in folder with indexed name
                output_filename = f"{ds_short_name}_progress{strategy_suffix}_{video_idx + 1:04d}.mp4"
                output_path = os.path.join(output_folder, output_filename)
            else:
                # Single video: use specified path or default
                if output_folder and not args.output.endswith(".mp4"):
                    output_path = os.path.join(output_folder, f"{ds_short_name}_progress{strategy_suffix}.mp4")
                elif not args.output.endswith(".mp4"):
                    output_path = f"{ds_short_name}_progress{strategy_suffix}_{args.output}.mp4"
                else:
                    # If user provided .mp4 extension, preserve directory and insert short name in filename
                    output_dir = os.path.dirname(args.output)
                    base_filename = os.path.basename(args.output)
                    base_name = os.path.splitext(base_filename)[0]
                    if output_dir:
                        output_path = os.path.join(
                            output_dir, f"{ds_short_name}_progress{strategy_suffix}_{base_name}.mp4"
                        )
                    else:
                        output_path = f"{ds_short_name}_progress{strategy_suffix}_{base_name}.mp4"

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
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Successful: {num_successful}/{args.num_videos}")
    print(f"  Failed: {num_failed}/{args.num_videos}")
    if args.num_videos > 1 and num_successful > 0:
        print(f"  Output folder: {os.path.abspath(output_folder)}")
    print(f"{'=' * 60}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
