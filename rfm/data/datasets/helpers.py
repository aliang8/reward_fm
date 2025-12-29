from math import e
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import random
from enum import Enum

import numpy as np
import json
import torch
from rfm.utils.distributed import rank_0_print
from rfm.data.dataset_types import Trajectory


class DataGenStrat(Enum):
    """Enum for different data generation strategies used in preference generation."""

    REVERSE_PROGRESS = "reverse_progress"
    UNIFORM_SAMPLE = "uniform_sample"
    FORWARD_PROGRESS = "forward_progress"
    REWIND = "rewind"
    SUBOPTIMAL = "suboptimal"
    DIFFERENT_TASK = "different_task"
    DIFFERENT_TASK_INSTRUCTION = "different_task_instruction"
    PAIRED_HUMAN_ROBOT = "paired_human_robot"
    ROBOARENA_PARTIAL_SUCCESS = "roboarena_partial_success"


def load_dataset_success_percent(cutoff_file_path):
    """Load dataset-specific success percentage from file."""
    success_percent_dict = {}
    try:
        with open(cutoff_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and "," in line:
                    dataset_name, success_percent = line.split(",")
                    success_percent_dict[dataset_name.strip()] = float(success_percent.strip())
    except FileNotFoundError:
        print(f"Warning: Success cutoff file {cutoff_file_path} not found. Using default thresholds.")
    return success_percent_dict


def convert_continuous_to_discrete_bin(value: float, num_bins: int) -> int:
    """Convert a single continuous progress value in [0, 1] to a discrete bin [0, num_bins-1].

    Args:
        value: Single continuous progress value in [0, 1]
        num_bins: Number of discrete bins to use

    Returns:
        Discrete bin index in [0, num_bins-1]
    """
    return int(min(max(value, 0.0), 1.0) * (num_bins - 1))


def convert_continuous_to_discrete_bins(progress_values: Union[List[float], np.ndarray], num_bins: int) -> List[int]:
    """Convert continuous progress values in [0, 1] to discrete bins [0, num_bins-1].

    Args:
        progress_values: List or array of continuous progress values in [0, 1]
        num_bins: Number of discrete bins to use

    Returns:
        List of discrete bin indices in [0, num_bins-1]
    """
    if isinstance(progress_values, np.ndarray):
        progress_values = progress_values.tolist()
    return [convert_continuous_to_discrete_bin(p, num_bins) for p in progress_values]


def compute_success_labels(
    target_progress: List[float],
    data_source: Optional[str],
    dataset_success_percent: Optional[Dict[str, float]] = None,
    max_success: float = 1.0,
    quality_label: Optional[str] = None,
) -> List[float]:
    """
    Compute success labels from target_progress.

    Args:
        target_progress: List of progress values (floats between 0 and 1)
        data_source: Data source name (used to look up dataset-specific threshold)
        dataset_success_percent: Dictionary mapping data source names to max_success thresholds
        max_success: Default max_success threshold if data_source not in dataset_success_percent
        quality_label: Quality label of the trajectory ("failure", "suboptimal", "successful", etc.)

    Returns:
        List of success labels (1.0 for success, 0.0 for failure) for each frame
    """
    if target_progress is None or len(target_progress) == 0:
        return []

    # If trajectory is failure or suboptimal, return all 0s
    if quality_label is not None and quality_label.lower() in ("failure", "suboptimal", "failed"):
        return [0.0] * len(target_progress)

    # Get the threshold for this data source
    if data_source is not None and dataset_success_percent is not None:
        threshold = dataset_success_percent.get(
            data_source if isinstance(data_source, str) else str(data_source), max_success
        )
    else:
        threshold = max_success

    # Generate success labels: 1.0 for success (progress > threshold), 0.0 for failure
    success_labels = [1.0 if prog > threshold else 0.0 for prog in target_progress]

    return success_labels


def load_frames_from_npz(npz_filepath: str) -> np.ndarray:
    """Load frames on-demand from npz file.

    Args:
        npz_filepath: Path to the .npz file containing frames

    Returns:
        numpy array with shape (T, H, W, C) containing the video frames
    """
    if not npz_filepath:
        raise ValueError("npz_filepath is None or empty")

    # If path is relative, prepend RFM_PROCESSED_DATASETS_PATH
    if not os.path.isabs(npz_filepath):
        rfm_dataset_path = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        # HACK:
        rfm_dataset_path = rfm_dataset_path.replace("processed_datasets", "")
        if rfm_dataset_path:
            npz_filepath = os.path.join(rfm_dataset_path, npz_filepath)

    if not os.path.exists(npz_filepath):
        raise ValueError(f"NPZ file not found: {npz_filepath}")

    try:
        # Load frames from npz file
        with np.load(npz_filepath) as data:
            frames = data["frames"]
            # Verify the data structure
            if "shape" in data:
                expected_shape = tuple(data["shape"])
                if frames.shape != expected_shape:
                    rank_0_print(f"Warning: Loaded frames shape {frames.shape} doesn't match expected {expected_shape}")

            return frames
    except Exception as e:
        rank_0_print(f"Error loading frames from {npz_filepath}: {e}")
        raise RuntimeError(f"Failed to load frames from {npz_filepath}: {e}")


def load_embeddings_from_path(embeddings_path: str) -> torch.Tensor:
    """Load video embeddings from .pt file and return just the video embeddings."""
    if not embeddings_path:
        import ipdb

        ipdb.set_trace()
        raise ValueError(f"embeddings_path: {embeddings_path} is None or empty")

    # If path is relative, prepend RFM_PROCESSED_DATASETS_PATH
    if not os.path.isabs(embeddings_path):
        rfm_dataset_path = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        # HACK:
        rfm_dataset_path = rfm_dataset_path.replace("processed_datasets/", "")
        rfm_dataset_path = rfm_dataset_path.replace("processed_datasets", "")
        if rfm_dataset_path:
            embeddings_path = os.path.join(rfm_dataset_path, embeddings_path)

    with open(embeddings_path, "rb") as f:
        embeddings_data = torch.load(f, map_location="cpu")
    return embeddings_data


def pad_trajectory_to_max_frames_np(
    frames: np.ndarray, progress: List[float], max_frames: int, pad_from: str = "right"
) -> Tuple[np.ndarray, List[float]]:
    """Pad trajectory frames and progress to max_frames by repeating the first frame/progress if needed.

    Args:
        frames: Trajectory frames (numpy array)
        progress: Progress values (list of floats)
        max_frames: Target number of frames

    Returns:
        Tuple[np.ndarray, List[float]: (padded_frames, padded_progress)
    """
    current_frames = frames.shape[0]

    if current_frames >= max_frames:
        # No padding needed
        return frames, progress

    if pad_from == "left":
        pad_frame = frames[0:1]  # Keep the batch dimension
        pad_progress = progress[0]
    else:
        pad_frame = frames[-1:]
        pad_progress = progress[-1]

    # Calculate how many frames to pad
    frames_to_pad = max_frames - current_frames

    # Pad frames by repeating the first frame
    if pad_from == "left":
        padded_frames = np.concatenate([np.repeat(pad_frame, frames_to_pad, axis=0), frames], axis=0)
        padded_progress = [pad_progress] * frames_to_pad + progress
    else:
        padded_frames = np.concatenate([frames, np.repeat(pad_frame, frames_to_pad, axis=0)], axis=0)
        padded_progress = progress + [pad_progress] * frames_to_pad

    return padded_frames, padded_progress


def pad_trajectory_to_max_frames_torch(
    frames, progress: List[float], max_frames: int, pad_from: str = "right"
) -> Tuple[torch.Tensor, List[float]]:
    """Pad trajectory frames and progress to max_frames by repeating the first frame/progress if needed.

    Args:
        frames: PyTorch tensor
        progress: Progress values (list of floats)
        max_frames: Target number of frames

    Returns:
        Tuple[torch.Tensor, List[float]: (padded_frames, padded_progress)
    """
    current_frames = frames.shape[0]

    if current_frames >= max_frames:
        # No padding needed
        return frames, progress

    # Need to pad - repeat the first frame and first progress
    if pad_from == "left":
        pad_frame = frames[0:1]  # Keep the batch dimension
        pad_progress = progress[0]
    else:
        pad_frame = frames[-1:]
        pad_progress = progress[-1]

    # Calculate how many frames to pad
    frames_to_pad = max_frames - current_frames

    # Pad frames by repeating the first frame
    padding = pad_frame.repeat(frames_to_pad, 1)

    if pad_from == "left":
        padded_frames = torch.cat([padding, frames], dim=0)
        padded_progress = [pad_progress] * frames_to_pad + progress
    else:
        padded_frames = torch.cat([frames, padding], dim=0)
        padded_progress = progress + [pad_progress] * frames_to_pad

    return padded_frames, padded_progress


def linspace_subsample_frames(
    frames: np.ndarray, num_frames: int = 8, end_idx: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """Uniformly subsample frames from a trajectory and return the indices.

    This method takes the full trajectory (e.g., 64 frames) and uniformly subsamples
    num_frames from it. The first and last frames are always included.

    Args:
        frames: Full trajectory frames (N frames)
        num_frames: Number of frames to subsample (default: 8)
        end_idx: Optional end index to subsample up to (if None, uses total_frames - 1)

    Returns:
        Tuple[np.ndarray, List[int]: (subsampled_frames, subsampled_indices)
    """
    if hasattr(frames, "shape"):
        total_frames = frames.shape[0]
    else:
        total_frames = len(frames)

    if total_frames <= 0:
        return frames, []

    # Use end_idx if provided, otherwise use full trajectory
    if end_idx is not None:
        end_idx = min(end_idx, total_frames - 1)
        frames_to_subsample = frames[: end_idx + 1]
        effective_total = end_idx + 1
    else:
        frames_to_subsample = frames
        effective_total = total_frames

    if effective_total <= num_frames:
        # If we have fewer (or equal) frames than requested, return all frames
        indices = list(range(effective_total))
        return frames_to_subsample, indices

    # Special case: if num_frames == 1, always take the last frame
    if num_frames == 1:
        indices = [effective_total - 1]
        subsampled_frames = frames_to_subsample[indices]
        return subsampled_frames, indices

    # Evenly spaced indices from 0 to effective_total-1, inclusive
    indices_np = np.linspace(0, effective_total - 1, num_frames)
    indices = np.rint(indices_np).astype(int).tolist()

    # Enforce first and last explicitly
    indices[0] = 0
    indices[-1] = effective_total - 1

    # Ensure indices are strictly non-decreasing and within bounds
    for k in range(1, len(indices)):
        if indices[k] < indices[k - 1]:
            indices[k] = indices[k - 1]
        if indices[k] >= effective_total:
            indices[k] = effective_total - 1

    # Subsample frames
    subsampled_frames = frames_to_subsample[indices]

    return subsampled_frames, indices


def randomly_subsample_frames(
    frames: np.ndarray, num_frames: int = 8, seed: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """Randomly subsample frames from a trajectory and return the indices.

    This method takes the full trajectory and randomly selects num_frames from it.
    This is useful for creating diverse trajectory samples and avoiding bias
    towards specific frame patterns.

    Args:
        frames: Full trajectory frames
        num_frames: Number of frames to subsample (default: 8)
        seed: Random seed for reproducible sampling (default: None)

    Returns:
        Tuple[np.ndarray, List[int]: (subsampled_frames, subsampled_indices)
    """
    if hasattr(frames, "shape"):
        total_frames = frames.shape[0]
    else:
        total_frames = len(frames)

    if total_frames < num_frames:
        # If we have fewer frames than requested, return all frames
        indices = list(range(total_frames))
        return frames, indices

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Randomly sample indices without replacement
    indices = sorted(random.sample(range(total_frames), num_frames))

    # Subsample frames
    subsampled_frames = frames[indices]

    return subsampled_frames, indices


def subsample_segment_frames_with_middle(
    frames: np.ndarray,
    max_frames: int,
    start_idx: int,
    end_idx: int,
    middle_idx: Optional[int] = None,
    method: str = "linspace",
) -> tuple[np.ndarray, int, int, int, list[int]]:
    """Subsample frames following the path: start -> middle -> end.

    Logic:
    1. If middle_idx is None, set it to end_idx (simple start -> end path)
    2. Enumerate all frames from start -> middle
    3. Enumerate all frames from middle -> end
    4. Join the two segments together
    5. Apply random or linspace subsampling to the combined segment

    Args:
        frames: Input frames array
        max_frames: Maximum number of frames to subsample
        start_idx: Start index
        end_idx: End index
        middle_idx: Optional middle index. If None, set to end_idx (simple start -> end path)
        method: Subsampling method ("linspace" or "random")

    Returns:
        Tuple of (subsampled_frames, start_idx, middle_idx, end_idx, subsampled_indices)
        where subsampled_indices are indices into the original frames array, ordered as start -> middle -> end
    """
    num_frames_total = len(frames)

    # Ensure indices are valid (allow end_idx < start_idx for reverse progress)
    start_idx = max(0, min(start_idx, num_frames_total - 1))

    # Handle edge cases for max_frames == 1 or 2
    if max_frames == 1:
        # Just return the single frame at start_idx
        indices = [start_idx]
        subsampled_frames = frames[start_idx : start_idx + 1]
        return subsampled_frames, start_idx, None, None, indices

    end_idx = max(0, min(end_idx, num_frames_total - 1))

    if max_frames == 2:
        indices = [start_idx, end_idx]
        subsampled_frames = frames[indices]
        return subsampled_frames, start_idx, None, end_idx, indices

    # If middle_idx is not provided, set it to end_idx (simple start -> end path)
    if middle_idx is None:
        middle_idx = end_idx
    else:
        middle_idx = max(0, min(middle_idx, num_frames_total - 1))

    # Enumerate frames from start -> middle
    if start_idx <= middle_idx:
        # Forward: start -> middle
        segment1_indices = list(range(start_idx, middle_idx + 1))
    else:
        # Backward: start -> middle (going backwards)
        segment1_indices = list(range(middle_idx, start_idx + 1))
        segment1_indices.reverse()  # Descending order: [start_idx, start_idx-1, ..., middle_idx]

    # Enumerate frames from middle -> end
    if middle_idx <= end_idx:
        # Forward: middle -> end
        segment2_indices = list(range(middle_idx, end_idx + 1))
    else:
        # Backward: middle -> end (going backwards)
        segment2_indices = list(range(end_idx, middle_idx + 1))
        segment2_indices.reverse()  # Descending order: [middle_idx, middle_idx-1, ..., end_idx]

    # Join the two segments, removing duplicate middle
    if segment1_indices and segment2_indices:
        # Remove middle from segment2 since it's already in segment1
        segment2_indices = [idx for idx in segment2_indices if idx != middle_idx]
        combined_indices = segment1_indices + segment2_indices
    elif segment1_indices:
        combined_indices = segment1_indices
    elif segment2_indices:
        combined_indices = segment2_indices
    else:
        # All indices are the same
        combined_indices = [start_idx]

    # Extract the frames corresponding to combined_indices
    combined_frames = frames[combined_indices]

    if method == "random":
        subsampled_frames, relative_indices = randomly_subsample_frames(combined_frames, num_frames=max_frames)
    else:
        subsampled_frames, relative_indices = linspace_subsample_frames(combined_frames, num_frames=max_frames)

    # Map relative indices back to original frame indices
    actual_indices = [combined_indices[i] for i in relative_indices]

    return subsampled_frames, start_idx, middle_idx, end_idx, actual_indices


def convert_absolute_to_relative_progress(absolute_progress: List[float]) -> List[float]:
    """Convert absolute progress values to relative deltas.

    Args:
        absolute_progress: List of absolute progress values (cumulative)

    Returns:
        List of relative progress deltas where first element is 0.0
        and subsequent elements are deltas: progress[i] - progress[i-1]
    """
    if not absolute_progress:
        return []

    relative_progress = [0.0]
    for i in range(1, len(absolute_progress)):
        relative_progress.append(absolute_progress[i] - absolute_progress[i - 1])

    return relative_progress


def _compute_absolute_wrt_total_frames_progress(
    num_frames_total: int,
    subsampled_indices: List[int],
    success_cutoff: Optional[float] = None,
) -> List[float]:
    """Compute progress using absolute_wrt_total_frames method.

    Args:
        num_frames_total: Total number of frames in the original trajectory.
        subsampled_indices: Absolute indices into the original trajectory.
        success_cutoff: Optional success cutoff threshold.

    Returns:
        List of progress values: progress[i] = (subsampled_indices[i] + 1) / num_frames_total.
    """
    segment_progress: List[float] = []
    cutoff_index: Optional[int] = None
    if success_cutoff is not None and success_cutoff > 0:
        # Index of the first frame where progress exceeds the cutoff
        cutoff_index = int(success_cutoff * num_frames_total)

    for abs_idx in subsampled_indices:
        if cutoff_index is not None and abs_idx >= cutoff_index:
            # All frames after cutoff get 1.0 progress
            progress = 1.0
        else:
            progress = (abs_idx + 1) / num_frames_total
        segment_progress.append(progress)
    return segment_progress


def _compute_absolute_first_frame_progress(
    num_frames_total: int,
    subsampled_indices: List[int],
    success_cutoff: Optional[float] = None,
) -> List[float]:
    """Compute progress using absolute_first_frame method.

    Args:
        num_frames_total: Total number of frames in the original trajectory.
        subsampled_indices: Absolute indices into the original trajectory.
        success_cutoff: Optional success cutoff threshold.

    Returns:
        List of progress values: progress[i] = (subsampled_indices[i] - start_idx) / (num_frames_total - start_idx - 1),
        where start_idx is the first index in subsampled_indices.
    """
    if not subsampled_indices:
        return []

    # Get the start index (first index in the subsampled indices)
    start_idx = min(subsampled_indices)

    cutoff_index: Optional[int] = None
    if success_cutoff is not None and success_cutoff > 0:
        # Index of the first frame where progress exceeds the cutoff
        cutoff_index = int(success_cutoff * num_frames_total)

    segment_progress: List[float] = []
    for abs_idx in subsampled_indices:
        # Calculate relative position from start
        relative_pos = abs_idx - start_idx

        if cutoff_index is not None:
            # ensure denominator is at least 1 to avoid division by zero
            denominator = max(1, cutoff_index - start_idx - 1)
            # if it goes pass the cutoff, the progress will be set to 1
            segment_progress.append(min(1.0, relative_pos / denominator))
        else:
            # ensure denominator is at least 1 to avoid division by zero
            denominator = max(1, num_frames_total - start_idx - 1)
            # Normal progress calculation
            segment_progress.append(relative_pos / denominator)

    return segment_progress


def _compute_relative_first_frame_progress(
    num_frames_total: int,
    subsampled_indices: List[int],
    success_cutoff: Optional[float] = None,
) -> List[float]:
    """Compute progress using relative_first_frame method.

    Args:
        num_frames_total: Total number of frames in the original trajectory.
        subsampled_indices: Absolute indices into the original trajectory.
        success_cutoff: Optional success cutoff threshold.

    Returns:
        List of relative progress deltas: progress[0] = 0.0;
        progress[i] = (subsampled_indices[i] - subsampled_indices[i-1]) / (num_frames_total - start_idx).
    """
    # First compute absolute_first_frame progress
    absolute_progress = _compute_absolute_first_frame_progress(num_frames_total, subsampled_indices, success_cutoff)
    # Convert to relative deltas
    return convert_absolute_to_relative_progress(absolute_progress)


def compute_progress_from_segment(
    num_frames_total: int,
    frame_indices: List[int],
    progress_pred_type: str = "absolute_first_frame",
    success_cutoff: Optional[float] = None,
) -> List[float]:
    """Compute progress values given total frames and subsampled indices.

    Args:
        num_frames_total: Total number of frames in the original trajectory (before segmenting).
        frame_indices: Absolute indices into the original trajectory.
        progress_pred_type: Type of progress calculation:
            - "absolute_first_frame": progress[i] = (frame_indices[i] - start_idx) / (num_frames_total - start_idx - 1),
              where start_idx is the minimum index in frame_indices.
            - "relative_first_frame": progress[0] = 0.0; progress[i] = (frame_indices[i] - frame_indices[i-1]) / (num_frames_total - start_idx).
            - "absolute_wrt_total_frames": progress[i] = (frame_indices[i] + 1) / num_frames_total.
        success_cutoff: Optional success cutoff threshold.

    Returns:
        List of progress values based on the specified progress_pred_type.
    """
    if progress_pred_type == "absolute_wrt_total_frames":
        return _compute_absolute_wrt_total_frames_progress(num_frames_total, frame_indices, success_cutoff)
    elif progress_pred_type == "relative_first_frame":
        return _compute_relative_first_frame_progress(num_frames_total, frame_indices, success_cutoff)
    else:  # default: "absolute_first_frame"
        return _compute_absolute_first_frame_progress(num_frames_total, frame_indices, success_cutoff)


def create_rewind_trajectory(
    original_traj: Dict[str, Any],
    rewind_length: Optional[int] = None,
    max_frames: int = 8,
    use_embeddings: bool = False,
    progress_pred_type: str = "absolute_first_frame",
    success_cutoff: Optional[float] = None,
    dataset_success_percent: Optional[Dict[str, float]] = None,
    max_success: float = 0.95,
) -> Trajectory:
    """Create a suboptimal trajectory by rewinding the original trajectory.

    This method creates a trajectory that goes forward then rewinds back:
    1. Selects start index in the first half of the original trajectory
    2. Selects end index in the latter half of the original trajectory
    3. Picks a rewind index between start and end
    4. Creates a forward segment from start index to end-1 (avoiding repetition)
    5. Creates a rewind segment by reversing from end-2 back to rewind_point (completely avoiding repetition)
    6. Concatenates forward + rewind to create the full trajectory
    7. Applies uniform subsampling to get the final num_frames

    Works with both frames and embeddings automatically.

    Example:
    Original frames: [0, 1, 2, ... 63]
    Start index: 10
    End index: 30
    Rewind point: 25
    Rewind length: 5
    Forward frames: [10, 11, 12, ..., 28, 29] # we include the start index, but not the end index
    Rewind frames: [28, 27, 26, 25] # we include the rewind point, but not the last frame of the forward segment
    Combined frames: [10, 11, 12, ..., 28, 29, 28, 27, 26, 25]

    # Note: always start at 1, the denominator is (num_frames - start_idx)
    Forward progress: [1/54, 2/54, 3/54, ..., 29/54, 30/54]
    Rewind progress: [29/54, 28/54, 27/54, 26/54]
    Combined progress: [1/54, 2/54, 3/54, ..., 29/54, 29/54, 28/54, 27/54, 26/54]

    Args:
        original_traj: Original trajectory dictionary
        rewind_length: Number of frames to rewind (default: random 1 to max_frames)
    """
    if use_embeddings:
        # Load embeddings from .pt file
        embeddings = load_embeddings_from_path(original_traj["embeddings_path"])
        frames_data = embeddings["video_embeddings"]
        text_embedding = embeddings["text_embedding"]
    else:
        # Load frames from npz file
        frames_data = load_frames_from_npz(original_traj["frames"])
        text_embedding = None

    # Get the number of frames
    if hasattr(frames_data, "shape"):
        num_frames = frames_data.shape[0]  # Use shape[0] for numpy array
    else:
        num_frames = len(frames_data)

    start = num_frames // 2
    if num_frames == 0 or (start - 1) <= 0:
        return None

    # Step 1: Select start and end indices
    # Start index is in the first half of the trajectory
    start_idx = random.randint(0, start - 1)
    # End index is in the latter half of the trajectory
    end_idx = random.randint(num_frames // 2, num_frames)

    # Ensure we have enough frames between start and end
    attempts = 0
    max_attempts = 10
    while end_idx - start_idx < 5 and attempts < max_attempts:
        start_idx = random.randint(0, start - 1)
        end_idx = random.randint(num_frames // 2, num_frames)
        attempts += 1

    # If we still don't have enough frames after max attempts, return None and try a different strategy
    if end_idx - start_idx < 5:
        return None

    # Step 2: Select rewind index between start and end
    if rewind_length is None:
        # Pick rewind point randomly between start+1 and end-1
        # We want at least 1 frame forward and at least 1 frame rewind
        rewind_point = random.randint(start_idx + 1, end_idx - 1)
        rewind_length = end_idx - rewind_point - 1
    else:
        # Ensure rewind_length is valid
        max_rewind = end_idx - start_idx - 1
        if rewind_length >= max_rewind:
            rewind_length = max_rewind
        if rewind_length < 1:
            rewind_length = 1
        rewind_point = start_idx + rewind_length

    # Step 3: Extract forward segment
    forward_frames = frames_data[start_idx:end_idx]
    forward_indices = list(range(start_idx, end_idx))  # start to end-1

    # Step 4: Create rewind segment
    # end at rewind_point-1 because we want to include the first frame of rewind segment
    reverse_frames = frames_data[rewind_point : end_idx - 1]
    if use_embeddings and torch is not None:
        reverse_frames = torch.flip(reverse_frames, dims=[0])
    else:
        reverse_frames = reverse_frames[::-1]

    # Step 5: Combine forward and reverse segments
    if use_embeddings and torch is not None:
        combined_frames = torch.cat([forward_frames, reverse_frames], dim=0)
    elif isinstance(forward_frames, np.ndarray):
        # If frames are numpy arrays, use concatenate
        combined_frames = np.concatenate([forward_frames, reverse_frames], axis=0)
    else:
        # If frames are lists, use regular concatenation
        combined_frames = forward_frames + reverse_frames

    # Step 6: Calculate progress for each frame position in the combined trajectory
    # Determine if success cutoff affects this segment
    cutoff_index = None
    if success_cutoff is not None and success_cutoff > 0:
        # Index of the first frame where progress exceeds the cutoff
        cutoff_index = int(success_cutoff * num_frames)

    # Calculate progress based on type
    if progress_pred_type == "absolute_wrt_total_frames":
        # For absolute_wrt_total_frames, calculate progress as (absolute_idx + 1) / num_frames
        # cutoff_index is already calculated above

        # Forward segment: indices from start_idx to end_idx-1
        forward_progress_abs = []
        for i in range(len(forward_indices)):
            abs_idx = start_idx + i
            if cutoff_index is not None and abs_idx >= cutoff_index:
                progress = 1.0
            else:
                progress = (abs_idx + 1) / num_frames
            forward_progress_abs.append(progress)

        # Rewind segment: reverse the forward progress (but we need to map to actual indices)
        # The rewind segment goes from end_idx-2 down to rewind_point
        rewind_actual_indices = list(range(end_idx - 2, rewind_point - 1, -1))  # Reverse order
        rewind_progress_abs = []
        for idx in rewind_actual_indices:
            if cutoff_index is not None and idx >= cutoff_index:
                progress = 1.0
            else:
                progress = (idx + 1) / num_frames
            rewind_progress_abs.append(progress)
    else:
        # For absolute_first_frame and relative_first_frame, use the original logic
        # Step 6: Calculate absolute progress for each frame position in the combined trajectory
        forward_progress_abs = []
        denom_norm = max(1, (num_frames - start_idx - 1))
        denom_cut = None
        if cutoff_index is not None and cutoff_index > start_idx:
            denom_cut = max(1, (cutoff_index - start_idx - 1))

        for i in range(len(forward_indices)):  # 0 to len(forward_indices)-1
            current_abs_idx = start_idx + i
            if denom_cut is not None:
                if current_abs_idx < cutoff_index:
                    forward_progress_abs.append(i / denom_norm)
                else:
                    forward_progress_abs.append(min(1.0, i / denom_cut))
            else:
                forward_progress_abs.append(i / denom_norm)

        # Rewind progress is the reverse of forward progress
        rewind_progress_abs = forward_progress_abs[::-1][1 : rewind_length + 1]

    # Combine absolute progress values
    combined_progress_abs = forward_progress_abs + rewind_progress_abs

    subsampled_frames, subsampled_indices = linspace_subsample_frames(combined_frames, max_frames)
    subsampled_progress = [combined_progress_abs[idx] for idx in subsampled_indices]
    subsampled_frames_shape = subsampled_frames.shape

    if progress_pred_type == "relative_first_frame":
        subsampled_progress = convert_absolute_to_relative_progress(subsampled_progress)

    if use_embeddings:
        subsampled_frames, subsampled_progress = pad_trajectory_to_max_frames_torch(
            subsampled_frames, subsampled_progress, max_frames
        )
    else:
        subsampled_frames, subsampled_progress = pad_trajectory_to_max_frames_np(
            subsampled_frames, subsampled_progress, max_frames
        )

    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "rewind_point": rewind_point,
        "rewind_length": rewind_length,
        "subsampled_indices": subsampled_indices,
    }

    return create_trajectory_from_dict(
        original_traj,
        overrides={
            "frames": subsampled_frames if not use_embeddings else None,
            "frames_shape": subsampled_frames_shape,
            "video_embeddings": subsampled_frames if use_embeddings else None,
            "text_embedding": text_embedding,
            "quality_label": "rewound",
            "target_progress": subsampled_progress,
            "metadata": metadata,
        },
    )


def create_trajectory_from_dict(traj_dict: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Trajectory:
    """Create a Trajectory from a dictionary with optional field overrides.

    This helper function simplifies Trajectory creation by extracting common fields
    from a trajectory dictionary and allowing specific fields to be overridden.

    Args:
        traj_dict: Dictionary containing trajectory data (e.g., from dataset)
        overrides: Dictionary of field values to override (e.g., frames, target_progress)

    Returns:
        A new Trajectory instance
    """
    traj_data = {
        "id": traj_dict.get("id"),
        "task": traj_dict.get("task"),
        "lang_vector": traj_dict.get("lang_vector"),
        "data_source": traj_dict.get("data_source"),
        "quality_label": traj_dict.get("quality_label"),
        "is_robot": traj_dict.get("is_robot"),
        "partial_success": traj_dict.get("partial_success"),
    }

    if overrides:
        traj_data.update(overrides)

    return Trajectory.model_validate(traj_data)


def show_available_datasets():
    """Show which datasets are available in the cache."""
    # The preprocessing script now creates individual cache directories for each dataset/subset pair
    cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError(
            "RFM_PROCESSED_DATASETS_PATH environment variable not set. Please set it to the directory containing your processed datasets."
        )

    print("=" * 100)
    print("Available datasets:")

    # List all subdirectories (individual dataset caches)
    if os.path.exists(cache_dir):
        subdirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
        if subdirs:
            for subdir in sorted(subdirs):
                # Try to load dataset info
                info_file = os.path.join(cache_dir, subdir, "dataset_info.json")
                if os.path.exists(info_file):
                    with open(info_file) as f:
                        info = json.load(f)
                    dataset_path = info.get("dataset_path", "unknown")
                    subset = info.get("subset", "unknown")
                    trajectories = info.get("total_trajectories", 0)
                    print(f"   {dataset_path}/{subset}: {trajectories} trajectories")
        else:
            print("  ❌ No dataset caches found")
    else:
        print("  ❌ Cache directory does not exist")
    print("=" * 100)
