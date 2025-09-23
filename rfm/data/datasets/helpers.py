import os
import random
from enum import Enum

import numpy as np

from rfm.utils.logging import rank_0_print


class DataGenStrat(Enum):
    """Enum for different data generation strategies used in preference generation."""

    # Preference generation strategies
    REWIND_SAME_TASK = "rewind_same_task"
    SUBOPTIMAL_SAME_TASK = "suboptimal_same_task"
    DIFFERENT_TASK = "different_task"
    VIDEO_BINNED = "video_binned"

    # Evaluation-specific strategies
    CONFUSION_MATRIX = "confusion_matrix"
    WRONG_TASK_PREFERENCE = "wrong_task_preference"

    # General strategies
    SUBSAMPLE_TASK = "subsample_task"
    REWOUND = "rewound"
    DEFAULT = "default"


def load_frames_from_npz(npz_filepath: str) -> np.ndarray:
    """Load frames on-demand from npz file.

    Args:
        npz_filepath: Path to the .npz file containing frames

    Returns:
        numpy array with shape (T, H, W, C) containing the video frames
    """
    if not npz_filepath or not os.path.exists(npz_filepath):
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


def pad_trajectory_to_max_frames(
    frames: np.ndarray, progress: list[float], max_frames: int
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

    # Need to pad - repeat the first frame and first progress
    first_frame = frames[0:1]  # Keep the batch dimension
    first_progress = progress[0]

    # Calculate how many frames to pad
    frames_to_pad = max_frames - current_frames

    # Pad frames by repeating the first frame
    padded_frames = np.concatenate([np.repeat(first_frame, frames_to_pad, axis=0), frames], axis=0)

    # Pad progress by repeating the first progress value
    padded_progress = [first_progress] * frames_to_pad + progress

    return padded_frames, padded_progress


def linspace_subsample_frames(frames: np.ndarray, num_frames: int = 8) -> tuple[np.ndarray, list[int]]:
    """Uniformly subsample frames from a trajectory and return the indices.

    This method takes the full trajectory (e.g., 64 frames) and uniformly subsamples
    num_frames from it. The first and last frames are always included.
    The indices are returned so progress can be calculated correctly for rewind trajectories.

    Args:
        frames: Full trajectory frames (N frames)
        num_frames: Number of frames to subsample (default: 8)

    Returns:
        Tuple[np.ndarray, List[int]: (subsampled_frames, subsampled_indices)
    """
    if hasattr(frames, "shape"):
        total_frames = frames.shape[0]
    else:
        total_frames = len(frames)

    if total_frames <= 0:
        return frames, []

    if total_frames <= num_frames:
        # If we have fewer (or equal) frames than requested, return all frames
        indices = list(range(total_frames))
        return frames, indices

    # Evenly spaced indices from 0 to total_frames-1, inclusive
    indices_np = np.linspace(0, total_frames - 1, num_frames)
    indices = np.rint(indices_np).astype(int).tolist()

    # Enforce first and last explicitly
    indices[0] = 0
    indices[-1] = total_frames - 1

    # Ensure indices are strictly non-decreasing and within bounds
    for k in range(1, len(indices)):
        if indices[k] < indices[k - 1]:
            indices[k] = indices[k - 1]
        if indices[k] >= total_frames:
            indices[k] = total_frames - 1

    # Subsample frames
    subsampled_frames = frames[indices]

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


def subsample_frames_and_progress(frames: np.ndarray, max_frames: int) -> tuple[np.ndarray, list[float]]:
    """For trajectory, sample start and end indices to create a segment.

    This makes the progress calculation consistent with rewind trajectories.
    """
    num_frames_total = len(frames)

    # Select start and end indices for the chosen trajectory segment
    # Start index is in the first half of the trajectory
    start_idx = random.randint(0, num_frames_total // 2 - 1)
    # End index is in the latter 1/3 of the trajectory
    end = (2 * num_frames_total) // 3
    end_idx = random.randint(end, num_frames_total)

    # Ensure we have enough frames between start and end
    while end_idx - start_idx < 5:
        start_idx = random.randint(0, num_frames_total // 2 - 1)
        end_idx = random.randint(end, num_frames_total)

    # Extract the chosen segment
    segment_frames = frames[start_idx:end_idx]
    segment_indices = list(range(start_idx, end_idx))

    # Calculate progress for the full segment first
    segment_progress = []
    for i in range(len(segment_indices)):
        segment_progress.append((i + 1) / (num_frames_total - start_idx))

    # Randomly subsample the chosen trajectory segment to num_frames
    frames, indices = randomly_subsample_frames(segment_frames, max_frames)

    # Map the subsampled indices to the corresponding progress values from the full segment
    # The chosen_indices tell us which frames from the segment we're using
    progress = [segment_progress[idx] for idx in indices]

    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "subsampled_indices": indices,
    }
    return frames, progress, metadata


def create_rewind_trajectory(original_traj: dict, rewind_length: int | None = None, max_frames: int = 8) -> dict:
    """Create a suboptimal trajectory by rewinding the original trajectory.

    This method creates a trajectory that goes forward then rewinds back:
    1. Selects start index in the first half of the original trajectory
    2. Selects end index in the latter half of the original trajectory
    3. Picks a rewind index between start and end
    4. Creates a forward segment from start index to end-1 (avoiding repetition)
    5. Creates a rewind segment by reversing from end-2 back to rewind_point (completely avoiding repetition)
    6. Concatenates forward + rewind to create the full trajectory
    7. Applies uniform subsampling to get the final num_frames
    8. Calculates progress relative to start index but out of total 64 frames

    Example:
    Original frames: [0, 1, 2, ... 63]
    Start index: 10
    End index: 30
    Rewind point: 25
    Rewind length: 5
    Forward frames: [9, 10, 11, ..., 28, 29] # we include the start index, but not the end index
    Rewind frames: [28, 27, 26, 25] # we include the rewind point, but not the last frame of the forward segment
    Combined frames: [9, 10, 11, ..., 28, 29, 28, 27, 26, 25]

    # Note: always start at 1, the denominator is (num_frames - start_idx)
    Forward progress: [1/54, 2/54, 3/54, ..., 29/54, 30/54]
    Rewind progress: [29/54, 28/54, 27/54, 26/54]
    Combined progress: [1/54, 2/54, 3/54, ..., 29/54, 29/54, 28/54, 27/54, 26/54]

    # We then apply subsampling to get num_frames frames
    # We use linspace subsampling to get evenly spaced frames, including the first and last frame

    Args:
        original_traj: Original trajectory dictionary
        rewind_length: Number of frames to rewind (default: random 1 to max_frames)
    """
    # Load frames from npz file
    frames_data = load_frames_from_npz(original_traj["frames"])

    # Get the number of frames
    if hasattr(frames_data, "shape"):
        num_frames = frames_data.shape[0]  # Use shape[0] for numpy array
    else:
        num_frames = len(frames_data)

    # Step 1: Select start and end indices
    # Start index is in the first half of the trajectory
    start_idx = random.randint(0, num_frames // 2 - 1)
    # End index is in the latter half of the trajectory
    end_idx = random.randint(num_frames // 2, num_frames)

    # Ensure we have enough frames between start and end
    while end_idx - start_idx < 5:
        start_idx = random.randint(0, num_frames // 2 - 1)
        end_idx = random.randint(num_frames // 2, num_frames)

    # Step 2: Select rewind index between start and end
    if rewind_length is None:
        # Pick rewind point randomly between start+1 and end-1
        # We want at least 1 frame forward and at least 1 frame rewind
        rewind_point = random.randint(start_idx + 1, end_idx - 1)
        rewind_length = end_idx - rewind_point
    else:
        # Ensure rewind_length is valid
        max_rewind = end_idx - start_idx - 1
        if rewind_length >= max_rewind:
            rewind_length = max_rewind
        if rewind_length < 1:
            rewind_length = 1
        rewind_point = start_idx + rewind_length

    # Step 3: Extract forward segment
    # Does not include end index to avoid
    forward_frames = frames_data[start_idx:end_idx]
    forward_indices = list(range(start_idx, end_idx))  # start to end-1

    # Step 4: Create rewind segment
    # NOTE: progress is relative to start index
    # Example: If start=10, rewind_point=25, end=40 (assuming 64 total frames):
    # Forward: [10, 11, 12, ..., 38, 39] (start to end-1, avoiding repetition)
    # Forward progress: [1/54, 2/54, 3/54, ..., 29/54, 30/54]
    # Rewind: [38, 37, 36, ..., 26, 25] (end-2 back to rewind_point+1)
    # Rewind progress: [29/54, 28/54, 27/54, ...] (going backwards from where forward left off)
    # Combined: [10, 11, 12, ..., 38, 39, 38, 37, ..., 26, 25]
    # Combined progress: [1/54, 2/54, 3/54, ..., 30/54, 29/54, 28/54, ...]

    # start from end-2 because we don't want to include the last frame of forward segment
    # end at rewind_point-1 because we want to include the first frame of rewind segment
    reverse_frames = frames_data[end_idx - 2 : rewind_point - 1 : -1]

    # Step 5: Combine forward and reverse segments
    if isinstance(forward_frames, np.ndarray):
        # If frames are numpy arrays, use concatenate
        combined_frames = np.concatenate([forward_frames, reverse_frames], axis=0)
    else:
        # If frames are lists, use regular concatenation
        combined_frames = forward_frames + reverse_frames

    # Step 6: Calculate progress for each frame position in the combined trajectory
    # Progress should represent position within the selected segment, starting from 1/64
    forward_progress = []
    for i in range(len(forward_indices)):  # 0 to len(forward_indices)-1
        # Progress starts at 1/(num_frames - start_idx) for first frame, increments by 1/(num_frames - start_idx) for each frame
        forward_progress.append((i + 1) / (num_frames - start_idx))  # Progress: 1/64, 2/64, 3/64, ...

    rewind_progress = forward_progress[::-1][1:rewind_length]

    # Combine progress values
    combined_progress = forward_progress + rewind_progress

    # Step 7: Apply linspace subsampling to get final num_frames
    # Use linspace for rewound trajectories to get predictable, evenly spaced frames
    # must include the first and last frame
    subsampled_frames, subsampled_indices = linspace_subsample_frames(combined_frames, max_frames)

    # Step 8: Map the subsampled indices to the corresponding progress values
    # The subsampled_indices tell us which frames from the combined trajectory we're using
    subsampled_progress = [combined_progress[idx] for idx in subsampled_indices]

    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "rewind_point": rewind_point,
        "rewind_length": rewind_length,
        "subsampled_indices": subsampled_indices,
    }

    # Create new trajectory with rewind frames
    rewind_traj = original_traj.copy()
    rewind_traj["frames"] = subsampled_frames
    rewind_traj["frames_shape"] = subsampled_frames.shape
    rewind_traj["target_progress"] = subsampled_progress
    rewind_traj["metadata"] = metadata
    rewind_traj["quality_label"] = "rewound"
    return rewind_traj
