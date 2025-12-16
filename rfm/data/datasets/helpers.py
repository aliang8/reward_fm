from math import e
import os
import random
from enum import Enum

import numpy as np
import json

from rfm.utils.distributed import rank_0_print
from rfm.data.dataset_types import Trajectory

try:
    import torch
except ImportError:
    torch = None


class DataGenStrat(Enum):
    """Enum for different data generation strategies used in preference generation."""

    SUCCESSFUL = "successful"
    SUBSEQUENCE = "subsequence"
    REVERSE_PROGRESS = "reverse_progress"
    UNIFORM_SAMPLE = "uniform_sample"
    SUBOPTIMAL = "suboptimal"
    REWOUND = "rewound"
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


def compute_success_labels(
    target_progress: list[float],
    data_source: str | None,
    dataset_success_percent: dict[str, float] | None = None,
    max_success: float = 0.95,
) -> list[float]:
    """
    Compute success labels from target_progress.

    Args:
        target_progress: List of progress values (floats between 0 and 1)
        data_source: Data source name (used to look up dataset-specific threshold)
        dataset_success_percent: Dictionary mapping data source names to max_success thresholds
        max_success: Default max_success threshold if data_source not in dataset_success_percent

    Returns:
        List of success labels (1.0 for success, 0.0 for failure) for each frame
    """
    if target_progress is None or len(target_progress) == 0:
        return []

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
    frames: np.ndarray, progress: list[float], max_frames: int, pad_from: str = "right"
) -> tuple[np.ndarray, list[float]]:
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
    frames, progress: list[float], max_frames: int, pad_from: str = "right"
) -> tuple:
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
    frames: np.ndarray, num_frames: int = 8, end_idx: int | None = None
) -> tuple[np.ndarray, list[int]]:
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
    frames: np.ndarray, num_frames: int = 8, seed: int | None = None
) -> tuple[np.ndarray, list[int]]:
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


def subsample_segment_frames(
    frames: np.ndarray,
    max_frames: int,
    method: str = "linspace",
    perc_start: float = 0.5,
    perc_end: float = 2.0 / 3.0,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[np.ndarray, int, int, list[int]]:
    """Choose a random segment [start_idx, end_idx) and subsample frames.

    Returns subsampled frames along with (start_idx, end_idx, subsampled_indices).

    Args:
        frames: Input frames array
        max_frames: Maximum number of frames to subsample
        method: Subsampling method ("linspace" or "random")
        perc_start: Percentage for start bound (used if start_idx is None)
        perc_end: Percentage for end bound (used if end_idx is None)
        start_idx: Optional start index (if provided, overrides perc_start)
        end_idx: Optional end index (if provided, overrides perc_end)
    """
    num_frames_total = len(frames)

    # If we have fewer frames than max_frames, just return all frames
    if num_frames_total < max_frames:
        return frames, 0, num_frames_total, list(range(num_frames_total))

    # If start_idx and end_idx are provided, use them directly
    if start_idx is not None and end_idx is not None:
        # Ensure indices are valid
        start_idx = max(0, min(start_idx, num_frames_total - 1))
        end_idx = max(start_idx + 1, min(end_idx, num_frames_total))
    else:
        # Clamp percentages to valid ranges
        perc_start = max(0.0, min(1.0, perc_start))
        perc_end = max(0.0, min(1.0, perc_end))

        # Select start and end indices for the chosen trajectory segment
        start_bound = int(perc_start * num_frames_total)
        end_bound = int(perc_end * num_frames_total)

        # Ensure end_bound is at least start_bound + 1
        end_bound = max(end_bound, start_bound + 1)

        start_idx = random.randint(0, max(0, start_bound))
        end_idx = random.randint(end_bound, num_frames_total)

        # Ensure we have enough frames between start and end
        attempts = 0
        max_attempts = 10
        while end_idx - start_idx < 5 and attempts < max_attempts:
            start_idx = random.randint(0, max(0, start_bound))
            end_idx = random.randint(end_bound, num_frames_total)
            attempts += 1

        if end_idx - start_idx < 5:
            start_idx = 0
            end_idx = num_frames_total

    # Extract the chosen segment
    segment_frames = frames[start_idx:end_idx]

    # Subsample the chosen trajectory segment to max_frames
    if method == "random":
        subsampled_frames, indices = randomly_subsample_frames(segment_frames, max_frames)
    else:
        subsampled_frames, indices = linspace_subsample_frames(segment_frames, max_frames)

    return subsampled_frames, start_idx, end_idx, indices


def convert_absolute_to_relative_progress(absolute_progress: list[float]) -> list[float]:
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


def compute_progress_from_segment(
    num_frames_total: int,
    start_idx: int,
    end_idx: int,
    frame_indices: list[int],
    progress_pred_type: str = "absolute_first_frame",
    success_cutoff: float | None = None,
) -> list[float]:
    """Compute progress values given total frames, segment, and subsampled indices.

    Args:
        num_frames_total: Total number of frames in the original trajectory (before segmenting).
        start_idx: Start index (inclusive) of the selected segment within the original trajectory.
        end_idx: End index (exclusive) of the selected segment within the original trajectory.
        frame_indices: Indices into the segment array returned by subsampling (0-based in-segment indices).
        progress_pred_type: Type of progress calculation:
            - "absolute_first_frame": progress[i] = i / (num_frames_total - start_idx - 1), evaluated at each selected in-segment index.
            - "relative_first_frame": progress[0] = 0.0; progress[i] = (frame_indices[i] - frame_indices[i-1]) / (num_frames_total - start_idx).
            - "absolute_wrt_total_frames": progress[i] = (start_idx + frame_indices[i] + 1) / num_frames_total.

    Behavior:
        - absolute_first_frame: progress[i] = i / (num_frames_total - start_idx - 1), evaluated at each selected in-segment index.
        - relative_first_frame: progress[0] = 0.0; progress[i] = (frame_indices[i] - frame_indices[i-1]) / (num_frames_total - start_idx).
        - absolute_wrt_total_frames: progress[i] = (start_idx + frame_indices[i] + 1) / num_frames_total.
    """
    # Handle absolute_wrt_total_frames first (simplest case)
    if progress_pred_type == "absolute_wrt_total_frames":
        segment_progress = []
        cutoff_index = None
        if success_cutoff is not None and success_cutoff > 0:
            # Index of the first frame where progress exceeds the cutoff
            cutoff_index = int(success_cutoff * num_frames_total)
        
        for idx in frame_indices:
            # Calculate absolute index in original trajectory
            abs_idx = start_idx + idx
            if cutoff_index is not None and abs_idx >= cutoff_index:
                # All frames after cutoff get 1.0 progress
                progress = 1.0
            else:
                progress = (abs_idx + 1) / num_frames_total
            segment_progress.append(progress)
        return segment_progress

    # Calculate progress for the full segment first
    segment_len = end_idx - start_idx
    assert segment_len > 0, "Segment length must be greater than 0"

    cutoff_index = None
    if success_cutoff is not None and success_cutoff > 0:
        # Index of the first frame where progress exceeds the cutoff
        cutoff_index = int(success_cutoff * num_frames_total)

    # Calculate progress at each frame in the segment
    # This is absolute_first_frame progress
    segment_progress = []
    for i in range(segment_len):
        if cutoff_index is not None:
            # ensure denominator is at least 1 to avoid division by zero
            denominator = max(1, cutoff_index - start_idx - 1)
            # if it goes pass the cutoff, the progress will be set to 1
            segment_progress.append(min(1.0, i / denominator))
        else:
            # ensure denominator is at least 1 to avoid division by zero
            denominator = max(1, num_frames_total - start_idx - 1)
            # Normal progress calculation
            segment_progress.append(i / denominator)

    # Determine progress at subsampled indices
    segment_progress = [segment_progress[idx] for idx in frame_indices]

    if progress_pred_type == "relative_first_frame":
        # Convert absolute progress to relative deltas
        return convert_absolute_to_relative_progress(segment_progress)

    # Default: absolute_first_frame
    return segment_progress


def subsample_pairs_and_progress(frames, max_frames: int, progress_pred_type: str = "absolute_first_frame"):
    """Create pairwise frames for progress prediction.

    Constructs pairs (o_i, o_i+1), (o_i+1, o_i), (o_i, o_i+T), (o_i+T, o_i)
    where i is a random index and T is a random delta in number of frames.
    Randomly selects one of these 4 pairs.

    Args:
        frames: Full trajectory frames (can be numpy array or torch tensor)
        max_frames: Maximum number of frame pairs to generate (will be paired to 2*max_frames total)
        progress_pred_type: Type of progress prediction:
            - "absolute_first_frame" or "relative_first_frame": delta between frames / num_frames_total
            - "absolute_wrt_total_frames": absolute indices / num_frames_total for each frame in pair

    Returns:
        Tuple of (subsampled_frames, progress_list, metadata)
    """
    # Check if frames is a torch tensor
    is_torch = isinstance(frames, torch.Tensor)

    num_frames_total = len(frames)

    # Generate a single random index i
    i = random.randint(0, num_frames_total - 2)  # Ensure i+1 is valid

    # Generate a random delta T (between 1 and remaining frames)
    T = random.randint(1, num_frames_total - i - 1)

    # Define the 4 possible pairs
    possible_pairs = [
        ([i, i + 1], "forward_single"),  # (o_i, o_i+1)
        ([i + 1, i], "backward_single"),  # (o_i+1, o_i)
        ([i, i + T], "forward_delta"),  # (o_i, o_i+T)
        ([i + T, i], "backward_delta"),  # (o_i+T, o_i)
    ]

    # Randomly select one of the 4 pairs
    selected_indices, pair_type = random.choice(possible_pairs)

    # Extract the selected pair
    pair_frames = [frames[idx] for idx in selected_indices]
    pair_indices = selected_indices

    # Convert back to torch tensor if input was a torch tensor
    if is_torch:
        pair_frames = torch.stack(pair_frames)
    else:
        pair_frames = np.stack(pair_frames)

    # Calculate progress based on type
    if progress_pred_type == "absolute_wrt_total_frames":
        # For each frame in the pair, calculate progress as (idx + 1) / num_frames_total
        progress = [(idx + 1) / num_frames_total for idx in pair_indices]
    else:
        # For absolute_first_frame and relative_first_frame, use delta between frames
        # Calculate progress as a single number: delta between the two frames
        # For pairwise, we predict the delta/change between the frame pair
        delta_indices = pair_indices[1] - pair_indices[0]
        progress = [delta_indices / num_frames_total]  # make sure it is a list

    metadata = {
        "pair_indices": pair_indices,
        "sampling_strategy": "pairwise",
        "pair_type": pair_type,
        "i": i,
        "T": T,
    }
    return pair_frames, progress, metadata


def create_rewind_trajectory(
    original_traj: dict,
    rewind_length: int | None = None,
    max_frames: int = 8,
    use_embeddings: bool = False,
    progress_pred_type: str = "absolute_first_frame",
    success_cutoff: float | None = None,
    dataset_success_percent: dict[str, float] | None = None,
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

    # Compute success labels
    success_label = compute_success_labels(
        target_progress=subsampled_progress,
        data_source=original_traj["data_source"],
        dataset_success_percent=dataset_success_percent,
        max_success=max_success,
    )

    return Trajectory(
        frames=subsampled_frames if not use_embeddings else None,
        frames_shape=subsampled_frames_shape,
        video_embeddings=subsampled_frames if use_embeddings else None,
        text_embedding=text_embedding,
        task=original_traj["task"],
        lang_vector=original_traj["lang_vector"],
        data_source=original_traj["data_source"],
        quality_label="rewound",
        is_robot=original_traj["is_robot"],
        target_progress=subsampled_progress,
        partial_success=original_traj.get("partial_success"),
        success_label=success_label,
        metadata=metadata,
    )


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
