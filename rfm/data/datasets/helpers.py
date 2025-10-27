import os
import random
from enum import Enum

import numpy as np
import json

from rfm.utils.distributed import rank_0_print

try:
    import torch
except ImportError:
    torch = None


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
    if not npz_filepath:
        raise ValueError("npz_filepath is None or empty")
    
    # If path is relative, prepend RFM_PROCESSED_DATASETS_PATH
    if not os.path.isabs(npz_filepath):
        rfm_dataset_path = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        # HACK: 
        rfm_dataset_path = rfm_dataset_path.replace("processed_datasets/", "")
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


def load_embeddings_from_path(embeddings_path: str, embedding_type: str = "video_embeddings") -> torch.Tensor:
    """Load video embeddings from .pt file and return just the video embeddings."""
    if not embeddings_path:
        raise ValueError("embeddings_path is None or empty")

    # If path is relative, prepend RFM_PROCESSED_DATASETS_PATH
    if not os.path.isabs(embeddings_path):
        rfm_dataset_path = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        # HACK: 
        rfm_dataset_path = rfm_dataset_path.replace("processed_datasets/", "")
        if rfm_dataset_path:
            embeddings_path = os.path.join(rfm_dataset_path, embeddings_path)

    with open(embeddings_path, "rb") as f:
        embeddings_data = torch.load(f, map_location="cpu")
    return embeddings_data[embedding_type]


def pad_trajectory_to_max_frames_np(
    frames: np.ndarray, progress: list[float], max_frames: int, pad_from: str = "left"
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


def pad_trajectory_to_max_frames_torch(frames, progress: list[float], max_frames: int, pad_from: str = "left") -> tuple:
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


def subsample_frames_and_progress(
    frames: np.ndarray, max_frames: int, progress_pred_type: str = "absolute"
) -> tuple[np.ndarray, list[float]]:
    """Linear subsample frames and progress to max_frames"""
    # subsampled_frames, indices = linspace_subsample_frames(frames, max_frames)

    # progress = [(idx + 1) / len(frames) for idx in indices]

    # metadata = {
    #     "subsampled_indices": indices,
    # }

    # return subsampled_frames, progress, metadata

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
    # frames, indices = randomly_subsample_frames(segment_frames, max_frames)
    subsampled_frames, indices = linspace_subsample_frames(segment_frames, max_frames)

    if progress_pred_type == "absolute":
        progress = [segment_progress[idx] for idx in indices]
    else:
        # Calculate relative progress as delta between consecutive frames
        relative_progress = []
        for i in range(len(indices)):
            if i == 0:
                relative_progress.append(0.0)
            else:
                delta = indices[i] - indices[i - 1]
                relative_progress.append(delta / (num_frames_total - start_idx))
        progress = relative_progress

    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "subsampled_indices": indices,
    }
    return subsampled_frames, progress, metadata


def subsample_pairs_and_progress(
    frames, max_frames: int, progress_pred_type: str = "absolute"
):
    """Create pairwise frames for progress prediction.
    
    Constructs pairs (o_i, o_i+1), (o_i+1, o_i), (o_i, o_i+T), (o_i+T, o_i)
    where i is a random index and T is a random delta in number of frames.
    Randomly selects one of these 4 pairs.
    
    Args:
        frames: Full trajectory frames (can be numpy array or torch tensor)
        max_frames: Maximum number of frame pairs to generate (will be paired to 2*max_frames total)
        progress_pred_type: "absolute" or "relative" progress prediction
    
    Returns:
        Tuple of (subsampled_frames, progress_list, metadata)
    """    
    # Check if frames is a torch tensor
    is_torch = isinstance(frames, torch.Tensor)
    
    num_frames_total = len(frames)
    
    # Generate a single random index i
    i = random.randint(0, num_frames_total - 2)  # Ensure i+1 is valid
    
    # Generate a random delta T (between 1 and remaining frames)
    max_delta = min(10, num_frames_total - i - 1)  # Cap T at reasonable value
    T = random.randint(1, max_delta)
    
    # Define the 4 possible pairs
    possible_pairs = [
        ([i, i + 1], "forward_single"),      # (o_i, o_i+1)
        ([i + 1, i], "backward_single"),      # (o_i+1, o_i)
        ([i, i + T], "forward_delta"),        # (o_i, o_i+T)
        ([i + T, i], "backward_delta"),       # (o_i+T, o_i)
    ]
    
    # Randomly select one of the 4 pairs
    selected_indices, pair_type = random.choice(possible_pairs)
    
    # Extract the selected pair
    pair_frames = [frames[idx] for idx in selected_indices]
    pair_indices = selected_indices
    
    # Trim to max_frames if needed
    if len(pair_frames) > max_frames:
        pair_frames = pair_frames[:max_frames]
        pair_indices = pair_indices[:max_frames]
    
    # Convert back to torch tensor if input was a torch tensor
    if is_torch:
        pair_frames = torch.stack(pair_frames)
    
    # Calculate progress for each frame
    if progress_pred_type == "absolute":
        progress = [(idx + 1) / num_frames_total for idx in pair_indices]
    else:
        # Relative progress: delta between consecutive frames
        relative_progress = []
        for idx in range(len(pair_indices)):
            if idx == 0:
                relative_progress.append(0.0)
            else:
                delta = pair_indices[idx] - pair_indices[idx - 1]
                relative_progress.append(delta / num_frames_total)
        progress = relative_progress
    
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
    progress_pred_type: str = "absolute",
) -> dict:
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

    Works with both frames and embeddings automatically.

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
    if use_embeddings:
        # Load embeddings from .pt file
        frames_data = load_embeddings_from_path(original_traj["embeddings_path"])
    else:
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
    # Progress should represent position within the selected segment, starting from 1/64
    forward_progress = []
    for i in range(len(forward_indices)):  # 0 to len(forward_indices)-1
        if progress_pred_type == "absolute":
            # Progress starts at 1/(num_frames - start_idx) for first frame, increments by 1/(num_frames - start_idx) for each frame
            forward_progress.append((i + 1) / (num_frames - start_idx))  # Progress: 1/64, 2/64, 3/64, ...
        else:
            if i == 0:
                forward_progress.append(0.0)
            else:
                forward_progress.append(1 / (num_frames - start_idx))

    if progress_pred_type == "absolute":
        rewind_progress = forward_progress[::-1][1:rewind_length]
    else:
        rewind_progress = (np.array(forward_progress[::-1][1:rewind_length]) * -1).tolist()

    # Combine progress values
    combined_progress = forward_progress + rewind_progress

    # Step 7: Apply linspace subsampling to get final num_frames
    # Use linspace for rewound trajectories to get predictable, evenly spaced frames
    # must include the first and last frame
    subsampled_frames, subsampled_indices = linspace_subsample_frames(combined_frames, max_frames)

    # Step 8: Map the subsampled indices to the corresponding progress values
    # The subsampled_indices tell us which frames from the combined trajectory we're using
    if progress_pred_type == "absolute":
        subsampled_progress = [combined_progress[idx] for idx in subsampled_indices]
    else:
        # if relative, we need to sum the progress values between each indices to get relative progress
        subsampled_progress = [0.0]
        for start, end in zip(subsampled_indices[:-1], subsampled_indices[1:]):
            subsampled_progress.append(sum(combined_progress[start:end]))

    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "rewind_point": rewind_point,
        "rewind_length": rewind_length,
        "subsampled_indices": subsampled_indices,
    }

    # Create new trajectory with rewind frames/embeddings
    rewind_traj = original_traj.copy()

    if use_embeddings:
        # Store embeddings instead of frames
        rewind_traj["frames"] = subsampled_frames  # Store embeddings in frames field for rewind
        rewind_traj["frames_shape"] = subsampled_frames.shape
        # Keep the original embeddings_path for reference
    else:
        # Store frames normally
        rewind_traj["frames"] = subsampled_frames
        rewind_traj["frames_shape"] = subsampled_frames.shape

    rewind_traj["target_progress"] = subsampled_progress
    rewind_traj["metadata"] = metadata
    rewind_traj["quality_label"] = "rewound"
    return rewind_traj


def generate_success_labels(
    progress: list[float], min_success: float, max_success: float
) -> tuple[list[bool], list[int]]:
    """Generate success labels and mask based on progress values.

    Args:
        progress: List of progress values (floats between 0 and 1)
        min_success: Progress threshold below which success label is 0 (failure)
        max_success: Progress threshold above which success label is 1 (success)

    Returns:
        Tuple of (success_labels, success_label_mask):
            - success_labels: List of bools indicating success (True) or failure (False)
            - success_label_mask: List of ints (1=predict, 0=ignore) indicating which frames to predict
    """
    success_labels = []
    success_label_mask = []

    for prog in progress:
        if prog < min_success:
            # Below threshold: label as failure (0)
            success_labels.append(False)
            success_label_mask.append(1)  # Predict this frame
        elif prog > max_success:
            # Above threshold: label as success (1)
            success_labels.append(True)
            success_label_mask.append(1)  # Predict this frame
        else:
            # In between: don't predict
            success_labels.append(False)  # Placeholder value
            success_label_mask.append(0)  # Don't predict this frame

    return success_labels, success_label_mask


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
