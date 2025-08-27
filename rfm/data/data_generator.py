#!/usr/bin/env python3
"""
DataGenerator class for producing batches of data for RFM model training with three prediction heads:
1. Preference prediction: Predict whether o^1 or o^2 are preferred
2. Progress prediction: Predict progress for a single trajectory
3. Comparative scoring: Rank o^1 and o^2 against a reference trajectory o^ref

The generator allows controlling the ratio between different prediction types.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
import shutil
import os
from pathlib import Path
import torch
from rfm.data.batch_collator import PreferenceSample, SimilaritySample, BatchCollator, Trajectory
from rfm.data.vqa_batch_collator import ProgressSample, VQABatchCollator
from datasets import concatenate_datasets, Dataset
from rfm.utils.logging import rank_0_print
import json
from rfm.utils.logging import timer


class DataGenerator:
    """Data generator for producing batches of prediction data with controlled ratios."""

    def __init__(self, config, is_evaluation=False):
        """Initialize DataGenerator with configuration."""
        self.config = config
        self.is_evaluation = is_evaluation

        # Choose datasets based on whether this is for evaluation or training
        if is_evaluation and config.eval_datasets:
            self.datasets = config.eval_datasets
            self.subsets = config.eval_subsets
        else:
            self.datasets = config.train_datasets
            self.subsets = config.train_subsets

        self.force_reprocess = config.force_reprocess

        # Initialize dataset and index mappings
        self.dataset = None
        self.robot_trajectories = []
        self.human_trajectories = []
        self.optimal_by_task = {}
        self.suboptimal_by_task = {}
        self.quality_indices = {}
        self.task_indices = {}
        self.source_indices = {}

        self.preference_ratio = config.preference_ratio
        self.similarity_ratio = 1.0 - config.preference_ratio
        self.dataset_preference_ratio = config.dataset_preference_ratio
        self.preference_strategy_ratio: List[float] = config.preference_strategy_ratio

        # Show available datasets for debugging
        self.show_available_datasets()

        # Load trajectory dataset
        self._load_trajectory_dataset()

        # Initialize preference and similarity datasets
        self._load_preference_dataset()
        self._load_similarity_dataset()

        rank_0_print(f"DataGenerator initialized with {len(self.dataset)} total trajectories")
        rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}")
        rank_0_print(f"  Tasks: {len(self.task_indices)}")
        rank_0_print(f"  Quality labels: {len(self.quality_indices)}")
        rank_0_print(f"  Data sources: {len(self.source_indices)}")

    def _load_trajectory_dataset(self):
        """Load trajectory dataset using preprocessed index-based cache."""
        cache_dir = "./processed_datasets"
        cache_type = "evaluation" if self.is_evaluation else "training"

        # Check if preprocessed cache exists
        if os.path.exists(cache_dir) and not self.force_reprocess:
            rank_0_print(f"Found preprocessed cache at {cache_dir}, loading {cache_type} datasets...")
            self._load_preprocessed_cache(cache_dir, is_training=not self.is_evaluation)
            rank_0_print(
                f"Successfully loaded preprocessed {cache_type} datasets with {len(self.dataset)} trajectory indices"
            )
        else:
            # If no cache exists, we need to run the preprocessor first
            rank_0_print(f"No preprocessed cache found. Please run preprocess_datasets.py first to create the cache.")
            raise RuntimeError(
                f"Dataset preprocessing required. Please run:\n"
                "uv run scripts/preprocess_datasets.py\n"
                "This will create the necessary index-based cache for efficient data loading."
            )

    def _load_preprocessed_cache(self, cache_dir: str, is_training: bool = True):
        """Load the preprocessed cache with index mappings for specific dataset/subset pairs."""
        # Validate the subsets format
        if not self.subsets:
            raise ValueError("No subsets configured. Please check your config.")

        # Check which dataset/subset pairs are available
        available_datasets = []
        missing_datasets = []

        rank_0_print(f"🔍 Checking cache for {len(self.datasets)} dataset(s) with subsets:")
        for i, (dataset_path, dataset_subsets) in enumerate(zip(self.datasets, self.subsets)):
            rank_0_print(f"  📁 Dataset {i + 1}: {dataset_path}")

            # Handle both single subset (string) and multiple subsets (list)
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]
                rank_0_print(f"    📂 Single subset: {dataset_subsets[0]}")
            elif isinstance(dataset_subsets, list):
                rank_0_print(f"    📂 Multiple subsets: {len(dataset_subsets)} subset(s)")
                for j, subset in enumerate(dataset_subsets):
                    rank_0_print(f"      📋 Subset {j + 1}: {subset}")
            else:
                raise ValueError(
                    f"Invalid subset format for {dataset_path}: {type(dataset_subsets)}. Expected str or List[str]"
                )

            for subset in dataset_subsets:
                cache_key = f"{dataset_path}/{subset}"
                # The preprocessing script creates individual cache directories for each dataset/subset pair
                individual_cache_dir = os.path.join(cache_dir, cache_key.replace("/", "_").replace(":", "_"))

                if os.path.exists(individual_cache_dir):
                    info_file = os.path.join(individual_cache_dir, "dataset_info.json")
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, "r") as f:
                                info = json.load(f)

                            available_datasets.append((dataset_path, subset, individual_cache_dir))
                            rank_0_print(f"      ✅ Found cache: {individual_cache_dir}")
                        except:
                            rank_0_print(f"      ⚠️  Cache info file corrupted, skipping: {individual_cache_dir}")
                            continue
                    else:
                        rank_0_print(f"      ⚠️  No info file found, skipping: {individual_cache_dir}")
                        continue
                else:
                    missing_datasets.append((dataset_path, subset))
                    rank_0_print(f"      ❌ Missing cache: {individual_cache_dir}")

        # Warn about missing datasets
        if missing_datasets:
            rank_0_print(f"\n⚠️  Warning: The following configured dataset/subset pairs are not available in the cache:")
            for dataset_path, subset in missing_datasets:
                rank_0_print(f"    ❌ {dataset_path}/{subset}")
            rank_0_print(f"  Available dataset/subset pairs will be loaded, but some configured data may be missing.")

        if not available_datasets:
            raise RuntimeError(
                f"No configured dataset/subset pairs are available in the cache. "
                f"Please run preprocess_datasets.py to create the cache for: {self.datasets}"
            )

        rank_0_print(f"\n📊 Summary: {len(available_datasets)} available, {len(missing_datasets)} missing")

        # Load available datasets
        loaded_datasets = []
        combined_indices = {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": {},
            "suboptimal_by_task": {},
            "quality_indices": {},
            "task_indices": {},
            "source_indices": {},
        }

        offset = 0

        for dataset_path, subset, individual_cache_dir in available_datasets:
            rank_0_print(f"📂 Loading {dataset_path}/{subset} from {individual_cache_dir}")

            # Load the processed dataset
            dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
            if not os.path.exists(dataset_cache_dir):
                rank_0_print(f"  ⚠️  Warning: Processed dataset not found at {dataset_cache_dir}, skipping...")
                continue

            dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=True)
            loaded_datasets.append(dataset)

            # Load index mappings
            mappings_file = os.path.join(individual_cache_dir, "index_mappings.json")
            if os.path.exists(mappings_file):
                with open(mappings_file, "r") as f:
                    indices = json.load(f)

                # Adjust indices by adding offset and combine
                for key in combined_indices:
                    if key in indices:
                        if isinstance(indices[key], list):
                            # For list indices, add offset
                            combined_indices[key].extend([idx + offset for idx in indices[key]])
                        elif isinstance(indices[key], dict):
                            # For dict indices, add offset to values
                            for subkey, subindices in indices[key].items():
                                if subkey not in combined_indices[key]:
                                    combined_indices[key][subkey] = []
                                combined_indices[key][subkey].extend([idx + offset for idx in subindices])

            rank_0_print(f"  ✅ Loaded {len(dataset)} trajectories from {dataset_path}/{subset}")
            offset += len(dataset)

        if not loaded_datasets:
            raise RuntimeError("No datasets could be loaded from the cache")

        # Concatenate datasets if multiple
        if len(loaded_datasets) == 1:
            self.dataset = loaded_datasets[0]
        else:
            rank_0_print(f"🔗 Concatenating {len(loaded_datasets)} loaded datasets...")
            self.dataset = concatenate_datasets(loaded_datasets)

        # Store the combined index mappings
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]

        dataset_type = "training" if is_training else "evaluation"
        rank_0_print(f"✅ Loaded {len(self.dataset)} total trajectories from preprocessed {dataset_type} datasets")
        rank_0_print(
            f"  📊 Available dataset/subset pairs: {len(available_datasets)}/{len(missing_datasets) + len(available_datasets)}"
        )
        rank_0_print(f"  📊 Missing dataset/subset pairs: {len(missing_datasets)}")

    def show_available_datasets(self):
        """Show which datasets are available in the cache."""
        # The preprocessing script now creates individual cache directories for each dataset/subset pair
        cache_dir = "./processed_datasets"

        rank_0_print(f"=" * 100)
        rank_0_print(f"\n🔍 Available datasets in {cache_dir}:")

        # List all subdirectories (individual dataset caches)
        if os.path.exists(cache_dir):
            subdirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
            if subdirs:
                for subdir in sorted(subdirs):
                    # Try to load dataset info
                    info_file = os.path.join(cache_dir, subdir, "dataset_info.json")
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, "r") as f:
                                info = json.load(f)
                            dataset_path = info.get("dataset_path", "unknown")
                            subset = info.get("subset", "unknown")
                            trajectories = info.get("total_trajectories", 0)
                            rank_0_print(f"  ✅ {dataset_path}/{subset}: {trajectories} trajectories")
                        except:
                            rank_0_print(f"  📁 {subdir}: (info file corrupted)")
                    else:
                        rank_0_print(f"  📁 {subdir}: (no info file)")
            else:
                rank_0_print(f"  ❌ No dataset caches found")
        else:
            rank_0_print(f"  ❌ Cache directory does not exist")
        rank_0_print(f"=" * 100)

        # Show configured datasets with better formatting for the new format
        rank_0_print(f"\n⚙️  Configured datasets:")
        for i, (dataset_path, dataset_subsets) in enumerate(zip(self.datasets, self.subsets)):
            rank_0_print(f"  📋 Dataset {i + 1}: {dataset_path}")

            # Handle both single subset (string) and multiple subsets (list)
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]
                rank_0_print(f"    📂 Single subset: {dataset_subsets[0]}")
            elif isinstance(dataset_subsets, list):
                rank_0_print(f"    📂 Multiple subsets: {len(dataset_subsets)} subset(s)")
                for j, subset in enumerate(dataset_subsets):
                    rank_0_print(f"      📋 Subset {j + 1}: {subset}")
            else:
                rank_0_print(f"    ⚠️  Invalid format: {type(dataset_subsets)}")

        # Show summary
        total_subsets = sum(len(subsets) if isinstance(subsets, list) else 1 for subsets in self.subsets)
        rank_0_print(f"\n📊 Total: {len(self.datasets)} dataset(s), {total_subsets} subset(s)")
        rank_0_print(f"=" * 100)

    def _load_frames_from_npz(self, npz_filepath: str) -> np.ndarray:
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
                        rank_0_print(
                            f"Warning: Loaded frames shape {frames.shape} doesn't match expected {expected_shape}"
                        )

                return frames
        except Exception as e:
            rank_0_print(f"Error loading frames from {npz_filepath}: {e}")
            raise RuntimeError(f"Failed to load frames from {npz_filepath}: {e}")

    def _get_trajectory_frames(self, trajectory_idx: int) -> np.ndarray:
        """Get frames for a trajectory by index, loading from npz if needed.

        Args:
            trajectory_idx: Index of the trajectory in the dataset

        Returns:
            numpy array with shape (T, H, W, C) containing the video frames
        """
        trajectory = self.dataset[trajectory_idx]
        npz_filepath = trajectory.get("frames")

        if not npz_filepath:
            raise ValueError(f"No frames path found for trajectory {trajectory_idx}")

        return self._load_frames_from_npz(npz_filepath)

    def _pad_trajectory_to_max_frames(self, frames: np.ndarray, progress: List[float], max_frames: int) -> Tuple[np.ndarray, List[float]]:
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
        padded_frames = np.concatenate([frames, np.repeat(first_frame, frames_to_pad, axis=0)], axis=0)
        
        # Pad progress by repeating the first progress value
        padded_progress = progress + [first_progress] * frames_to_pad
        
        return padded_frames, padded_progress

    def _linspace_subsample_frames(self, frames: np.ndarray, num_frames: int = 8) -> Tuple[np.ndarray, List[int]]:
        """Linspace subsample frames from a trajectory and return the indices.
        
        This method takes the full trajectory and uses numpy linspace to get evenly
        distributed frame indices. This is useful for rewound trajectories where we
        want predictable, evenly spaced frames.
        
        Args:
            frames: Full trajectory frames
            num_frames: Number of frames to subsample (default: 8)
            
        Returns:
            Tuple[np.ndarray, List[int]: (subsampled_frames, subsampled_indices)
            
        Example:
            If we have 64 frames and want 8 frames:
            - Linspace indices: [0, 9, 18, 27, 36, 45, 54, 63]
            - Subsampled frames: frames[0], frames[9], frames[18], etc.
        """
        if hasattr(frames, "shape"):
            total_frames = frames.shape[0]
        else:
            total_frames = len(frames)
            
        if total_frames < num_frames:
            # If we have fewer frames than requested, return all frames
            indices = list(range(total_frames))
            return frames, indices
            
        # Use numpy linspace to get evenly distributed indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Subsample frames
        subsampled_frames = frames[indices]
        
        return subsampled_frames, indices.tolist()

    def _uniformly_subsample_frames(self, frames: np.ndarray, num_frames: int = 8) -> Tuple[np.ndarray, List[int]]:
        """Uniformly subsample frames from a trajectory and return the indices.
        
        This method takes the full trajectory (64 frames) and uniformly subsamples
        num_frames from it. The indices are returned so progress can be calculated
        correctly for rewind trajectories.
        
        Args:
            frames: Full trajectory frames (64 frames)
            num_frames: Number of frames to subsample (default: 8)
            
        Returns:
            Tuple[np.ndarray, List[int]: (subsampled_frames, subsampled_indices)
            
        Example:
            If we have 64 frames and want 8 frames:
            - Original progress: [1/64, 2/64, 3/64, ..., 64/64]
            - Subsampled indices: [0, 9, 18, 27, 36, 45, 54, 63]
            - Subsampled frames: frames[0], frames[9], frames[18], etc.
        """
        if hasattr(frames, "shape"):
            total_frames = frames.shape[0]
        else:
            total_frames = len(frames)
            
        if total_frames < num_frames:
            # If we have fewer frames than requested, return all frames
            indices = list(range(total_frames))
            return frames, indices
            
        # Calculate step size for uniform sampling
        step_size = (total_frames - 1) / (num_frames - 1)
        
        # Generate indices for uniform sampling
        indices = []
        for i in range(num_frames):
            if i == num_frames - 1:
                # Ensure we include the last frame
                indices.append(total_frames - 1)
            else:
                indices.append(int(round(i * step_size)))
        
        # Subsample frames
        subsampled_frames = frames[indices]
        
        return subsampled_frames, indices

    def _create_rewind_trajectory(self, original_traj: Dict, rewind_length: Optional[int] = None) -> Dict:
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

        Args:
            original_traj: Original trajectory dictionary
            rewind_length: Number of frames to rewind (default: random 1 to max_frames)
        """
        # Load frames from npz file
        frames_data = self._load_frames_from_npz(original_traj["frames"])

        # Get the number of frames
        if hasattr(frames_data, "shape"):
            num_frames = frames_data.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames_data)

        if num_frames < 4:
            # If trajectory is too short, just return the original
            return original_traj

        # Step 1: Select start and end indices
        # Start index is in the first half of the trajectory
        start_idx = random.randint(0, num_frames // 2 - 1)
        # End index is in the latter half of the trajectory
        end_idx = random.randint(num_frames // 2, num_frames)
        
        # Ensure we have enough frames between start and end
        while end_idx - start_idx < 3:
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
        reverse_frames = frames_data[end_idx-2:rewind_point-1:-1]

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
        num_frames_to_sample = getattr(self.config, 'max_frames', 8)
        subsampled_frames, subsampled_indices = self._linspace_subsample_frames(combined_frames, num_frames_to_sample)

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

    def _create_video_binned_trajectory(self, original_traj: Dict, num_bins: int = 10) -> Tuple[Dict, Dict]:
        """Create a preference sample by splitting a video into temporal bins and sampling from different bins.
        
        This strategy creates preference samples by:
        1. Splitting the original video into N temporal bins (e.g., 4 bins for a 32-frame video)
        2. Randomly selecting two different bins from the same video
        3. Creating a preference sample where one bin represents progress and the other represents regression
        
        **Example:**
        ```
        Original video: 32 frames
        Bins: [0-7], [8-15], [16-23], [24-31]
        
        Strategy 1: Compare early progress vs late progress
        - Chosen: frames [16-23] (bin 2, middle progress)
        - Rejected: frames [0-7] (bin 0, early progress)
        
        Strategy 2: Compare progress vs regression
        - Chosen: frames [24-31] (bin 3, final progress)
        - Rejected: frames [16-23] (bin 2, middle progress, but shown in reverse)
        
        Strategy 3: Compare adjacent bins with different progress
        - Chosen: frames [8-15] (bin 1, early-mid progress)
        - Rejected: frames [0-7] (bin 0, early progress)
        ```
        
        **Benefits:**
        - Teaches the model to recognize temporal progress within the same task
        - Helps distinguish between early, middle, and late stages of task completion
        - Creates diverse preference pairs from the same video without external data
        - Useful for learning fine-grained temporal dynamics and progress indicators
        
        Args:
            original_traj: Original trajectory dictionary containing video frames
            num_bins: Number of temporal bins to split the video into (default: 10)
            
        Returns:
            Tuple[Dict, Dict]: (chosen_trajectory, rejected_trajectory) where both are modified
            trajectories with frames from different bins and updated metadata
            
        Raises:
            ValueError: If video is too short to create meaningful bins
            RuntimeError: If video binning fails for any reason
        """
        # Load frames from npz file
        frames_data = self._load_frames_from_npz(original_traj["frames"])
        
        # Get the number of frames
        if hasattr(frames_data, "shape"):
            num_frames = frames_data.shape[0]
        else:
            num_frames = len(frames_data)
            
        if num_frames < num_bins * 2:
            raise ValueError(f"Video too short ({num_frames} frames) to create {num_bins} meaningful bins")
            
        # Calculate bin size and boundaries
        bin_size = num_frames // num_bins
        bin_boundaries = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else num_frames
            bin_boundaries.append((start, end))
            
        # Randomly select two different bins
        bin_indices = list(range(num_bins))
        chosen_bin_idx = random.choice(bin_indices)
        bin_indices.remove(chosen_bin_idx)
        rejected_bin_idx = random.choice(bin_indices)
        
        # Extract frames from the chosen bin (this will be the "chosen" trajectory)
        chosen_start, chosen_end = bin_boundaries[chosen_bin_idx]
        chosen_frames = frames_data[chosen_start:chosen_end]
        
        # Extract frames from the rejected bin (this will be the "rejected" trajectory)
        rejected_start, rejected_end = bin_boundaries[rejected_bin_idx]
        rejected_frames = frames_data[rejected_start:rejected_end]
        
        # Apply uniform subsampling to both bins to ensure consistent frame counts
        # Use uniform subsampling for real trajectories (not rewound)
        num_frames_to_sample = getattr(self.config, 'max_frames', 8)
        chosen_frames, chosen_indices = self._uniformly_subsample_frames(chosen_frames, num_frames_to_sample)
        rejected_frames, rejected_indices = self._uniformly_subsample_frames(rejected_frames, num_frames_to_sample)
        
        # Calculate progress for each bin relative to the original trajectory
        chosen_progress = [chosen_start + idx for idx in chosen_indices]
        chosen_progress = [p / (len(frames_data) - 1) for p in chosen_progress]
        
        rejected_progress = [rejected_start + idx for idx in rejected_indices]
        rejected_progress = [p / (len(frames_data) - 1) for p in rejected_progress]
        
        # Store original frame positions for reference
        chosen_original_positions = [chosen_start + idx for idx in chosen_indices]
        rejected_original_positions = [rejected_start + idx for idx in rejected_indices]
        
        # Create the chosen trajectory (from chosen bin)
        chosen_traj = original_traj.copy()
        chosen_traj["frames"] = chosen_frames
        chosen_traj["frames_shape"] = chosen_frames.shape
        chosen_traj["id"] = f"{original_traj['id']}_bin_{chosen_bin_idx}_chosen"
        chosen_traj["quality_label"] = "video_binned_chosen"
        chosen_traj["metadata"] = chosen_traj.get("metadata", {}).copy()
        chosen_traj["metadata"]["video_binned_generated"] = True
        chosen_traj["metadata"]["original_traj_id"] = original_traj["id"]
        chosen_traj["metadata"]["chosen_bin_idx"] = chosen_bin_idx
        chosen_traj["metadata"]["bin_progress"] = chosen_progress
        chosen_traj["metadata"]["bin_frames"] = (chosen_start, chosen_end)
        chosen_traj["metadata"]["num_bins"] = num_bins
        chosen_traj["metadata"]["bin_size"] = bin_size
        chosen_traj["metadata"]["subsampled_generated"] = True
        chosen_traj["metadata"]["subsampled_progress"] = chosen_progress
        chosen_traj["metadata"]["num_frames_subsampled"] = num_frames_to_sample
        chosen_traj["metadata"]["original_frame_positions"] = chosen_original_positions
        
        # Create the rejected trajectory (from rejected bin)
        rejected_traj = original_traj.copy()
        rejected_traj["frames"] = rejected_frames
        rejected_traj["frames_shape"] = rejected_frames.shape
        rejected_traj["id"] = f"{original_traj['id']}_bin_{rejected_bin_idx}_rejected"
        rejected_traj["quality_label"] = "video_binned_rejected"
        rejected_traj["metadata"] = rejected_traj.get("metadata", {}).copy()
        rejected_traj["metadata"]["video_binned_generated"] = True
        rejected_traj["metadata"]["original_traj_id"] = original_traj["id"]
        rejected_traj["metadata"]["rejected_bin_idx"] = rejected_bin_idx
        rejected_traj["metadata"]["bin_progress"] = rejected_progress
        rejected_traj["metadata"]["bin_frames"] = (rejected_start, rejected_end)
        rejected_traj["metadata"]["num_bins"] = num_bins
        rejected_traj["metadata"]["bin_size"] = bin_size
        rejected_traj["metadata"]["subsampled_generated"] = True
        rejected_traj["metadata"]["subsampled_progress"] = rejected_progress
        rejected_traj["metadata"]["num_frames_subsampled"] = num_frames_to_sample
        rejected_traj["metadata"]["original_frame_positions"] = rejected_original_positions
        
        return chosen_traj, rejected_traj

    def _create_preference_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            raise ValueError("No preferences loaded from dataset")

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        preference = random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        raise NotImplementedError("Preference sample creation from dataset not yet implemented")

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_print("No preference dataset provided, will use random sampling for preferences")
        return

    def _load_similarity_dataset(self):
        """Load the similarity dataset if provided."""
        # For now, we'll use empty similarity dataset
        # This can be updated later if needed
        rank_0_print("No similarity dataset provided, will use random sampling for similarities")
        return

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.

        This method implements three different strategies for generating rejected trajectories
        to create diverse and robust preference learning data:

        **Strategy 1: Rewind Same Task**
        - Creates a suboptimal trajectory by rewinding the chosen trajectory
        - Same task, different trajectory ID
        - Good for learning task-specific failure modes and temporal dynamics

        **Strategy 2: Suboptimal/Failure Same Task**
        - Uses existing suboptimal/failure trajectories from the same task
        - Same task, different trajectory ID
        - Good for learning from real failure examples and task-specific suboptimal patterns

        **Strategy 3: Different Task**
        - Uses trajectories from completely different tasks
        - Different task, can be chosen or suboptimal
        - Good for learning cross-task generalization and what makes trajectories "good"
          across different contexts

        The strategy ratios are controlled by config.preference_strategy_ratio
        with default [0.8, 0.1, 0.1] for [rewind_same_task, suboptimal_same_task, different_task].

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata
        """

        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_preference_sample_from_dataset()
            else:
                return self._create_preference_sample_with_strategies()

    def _subsample_frames_and_progress(self, frames: np.ndarray, max_frames: int) -> Tuple[np.ndarray, List[float]]:
        # For chosen trajectory, sample start and end indices to create a segment
        # This makes the progress calculation consistent with rewind trajectories
        num_frames_total = len(frames)
        
        # Select start and end indices for the chosen trajectory segment
        # Start index is in the first half of the trajectory
        start_idx = random.randint(0, num_frames_total // 2 - 1)
        # End index is in the latter half of the trajectory
        end_idx = random.randint(num_frames_total // 2, num_frames_total)
        
        # Ensure we have enough frames between start and end
        while end_idx - start_idx < 3:
            start_idx = random.randint(0, num_frames_total // 2 - 1)
            end_idx = random.randint(num_frames_total // 2, num_frames_total)
        
        # Extract the chosen segment
        segment_frames = frames[start_idx:end_idx]
        segment_indices = list(range(start_idx, end_idx))
        
        # Calculate progress for the full segment first (like forward indices in rewind)
        # Progress should represent position within the selected segment, starting from 1/64
        segment_progress = []
        for i in range(len(segment_indices)):
            segment_progress.append((i + 1) / (num_frames_total - start_idx))
        
        # Uniformly subsample the chosen trajectory segment to num_frames 
        frames, indices = self._uniformly_subsample_frames(segment_frames, self.config.max_frames)
        
        # Map the subsampled indices to the corresponding progress values from the full segment
        # The chosen_indices tell us which frames from the segment we're using
        progress = [segment_progress[idx] for idx in indices]

       
        # Ensure both trajectories have exactly max_frames by padding if needed
        # Pad by repeating the first frame and first progress value
        frames, progress = self._pad_trajectory_to_max_frames(
            frames, progress, self.config.max_frames
        )

        metadata = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "subsampled_indices": indices,
        }
        return frames, progress, metadata

    def _create_preference_sample_with_strategies(self) -> PreferenceSample:
        """Create a preference prediction sample using various rejected trajectory generation strategies.

        This method implements four different strategies for generating rejected trajectories
        to create diverse and robust preference learning data. The strategy is chosen
        probabilistically according to self.preference_strategy_ratio.

        **Strategy 1: Rewind Same Task**
        - Creates a suboptimal trajectory by rewinding the chosen trajectory
        - Same task, different trajectory ID, artificially generated suboptimal behavior
        - Good for learning task-specific failure modes and temporal dynamics
        - Example: Forward progress [0→1→2→3] + rewind [2→1] = [0→1→2→3→2→1]

        **Strategy 2: Suboptimal/Failure Same Task**
        - Uses existing suboptimal/failure trajectories from the same task
        - Same task, different trajectory ID, real failure examples
        - Good for learning from actual failure patterns and task-specific suboptimal behaviors
        - Example: Compare successful "open door" vs failed "open door" attempts

        **Strategy 3: Different Task**
        - Uses trajectories from completely different tasks
        - Different task, can be chosen or suboptimal
        - Good for learning cross-task generalization and what makes trajectories "good"
          across different contexts
        - Example: Compare "open door" (successful) vs "press button" (successful)

        **Strategy 4: Video Binned**
        - Splits a single video into temporal bins and compares different bins
        - Same task, same video, different temporal segments
        - Good for learning temporal progress within the same task and fine-grained
          temporal dynamics
        - Example: Compare early progress [frames 0-7] vs late progress [frames 24-31]

        **Fallback Behavior:**
        If any strategy fails (e.g., no suboptimal trajectories available, video too short),
        the system automatically falls back to the rewind strategy to ensure robust
        data generation.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata

        Raises:
            ValueError: If no chosen trajectories are available for preference generation
            RuntimeError: If all strategies fail and fallback rewind also fails
        """

        # Use preprocessed chosen trajectories from index maps
        if not self.optimal_by_task:
            raise ValueError("No chosen trajectories found for preference generation")

        # Get a random task and chosen trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))

        optimal_indices = self.optimal_by_task[task_name]
        while not optimal_indices:
            task_name = random.choice(list(self.optimal_by_task.keys()))
            optimal_indices = self.optimal_by_task[task_name]

        chosen_idx = random.choice(optimal_indices)
        chosen_traj = self.dataset[chosen_idx]

        # Initialize variables for strategy selection
        rejected_traj = None
        strategy_used = None
        
        if random.random() < self.preference_strategy_ratio[0]:
            # Strategy 1: Use rewind-generated suboptimal trajectory from same task
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"
            
        elif random.random() < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1]:
            # Strategy 2: Use random suboptimal trajectory from same task
            same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
            same_task_suboptimal = [
                self.dataset[idx]
                for idx in same_task_suboptimal_indices
                if self.dataset[idx]["id"] != chosen_traj["id"]
            ]
            if same_task_suboptimal:
                rejected_traj = random.choice(same_task_suboptimal)
                strategy_used = "suboptimal_same_task"
                
        elif random.random() < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1] + self.preference_strategy_ratio[2]:
            # Strategy 3: Use trajectory from different task (can be chosen or suboptimal)
            other_tasks = [task for task in self.optimal_by_task.keys() if task != chosen_traj["task"]]
            if other_tasks:
                other_task = random.choice(other_tasks)
                other_task_indices = self.optimal_by_task[other_task]
                if other_task_indices:
                    other_idx = random.choice(other_task_indices)
                    other_traj = self.dataset[other_idx]
                    # Check if it's not the same trajectory
                    if other_traj["id"] != chosen_traj["id"]:
                        rejected_traj = other_traj
                        strategy_used = "different_task"
                        
        else:
            # Strategy 4: Create preference sample from different bins of the same video
            try:
                chosen_traj, rejected_traj = self._create_video_binned_trajectory(chosen_traj, num_bins=self.config.num_bins)
                strategy_used = "video_binned"
            except Exception as e:
                rank_0_print(f"Video binning failed: {e}, will fall back to rewind")
        
        # Fallback: If any strategy failed to produce a rejected trajectory, use rewind
        if rejected_traj is None:
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"

        # ===============================================================
        # Subsample the chosen trajectory to max_frames
        # ===============================================================
        if isinstance(chosen_traj["frames"], str):
            chosen_traj["frames"] = self._load_frames_from_npz(chosen_traj["frames"])

        chosen_frames, chosen_progress, chosen_metadata = self._subsample_frames_and_progress(chosen_traj["frames"], self.config.max_frames)

        # ===============================================================
        # Subsample the rejected trajectory to max_frames
        # ===============================================================

        if isinstance(rejected_traj["frames"], str):
            rejected_traj["frames"] = self._load_frames_from_npz(rejected_traj["frames"])

        if "rewind" not in strategy_used:
            # try subsampling the rejected trajectory 
            rejected_frames, rejected_progress, rejected_metadata = self._subsample_frames_and_progress(rejected_traj["frames"], self.config.max_frames)
        else:
            rejected_frames = rejected_traj["frames"]
            rejected_progress = rejected_traj["target_progress"]
            rejected_metadata = rejected_traj["metadata"]
        
        # Create preference sample structure
        sample = PreferenceSample(
            # Create Trajectory objects for chosen and rejected
            chosen_trajectory=Trajectory(
                frames=chosen_frames,
                frames_shape=chosen_frames.shape,
                id=chosen_traj["id"],
                task=chosen_traj["task"],
                lang_vector=chosen_traj["lang_vector"],
                data_source=chosen_traj["data_source"],
                quality_label=chosen_traj.get("quality_label"),
                is_robot=chosen_traj["is_robot"],
                target_progress=chosen_progress,
                data_gen_strategy="subsample_task",
                metadata=chosen_metadata
            ),
            rejected_trajectory=Trajectory(
                frames=rejected_frames,
                frames_shape=rejected_frames.shape,
                id=rejected_traj["id"],
                task=rejected_traj["task"],
                lang_vector=rejected_traj["lang_vector"],
                data_source=rejected_traj["data_source"],
                quality_label=rejected_traj["quality_label"],
                is_robot=rejected_traj["is_robot"],
                target_progress=rejected_progress,
                data_gen_strategy=strategy_used,
                metadata=rejected_metadata
            ),
        )
        return sample

    def _create_similarity_sample(self) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.

        Two modes (50/50 split):
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
        """

        # Randomly choose between rewind mode and optimal/suboptimal mode
        use_rewind_mode = random.choice([True, False])

        if use_rewind_mode:
            return self._create_rewind_similarity_sample()
        else:
            return self._create_optimal_similarity_sample()

    def _create_rewind_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using rewind logic.

        Rules:
        - traj_sim is rewound trajectory from same task as o^ref (different trajectory)
        - traj_diff MUST be from different task
        - traj_sim is preferred over traj_diff relative to o^ref

        Minimal requirements:
        - At least 2 trajectories in reference task (for o^ref and traj_sim rewind)
        - At least 2 tasks (for traj_diff from different task)
        """

        # Get available tasks
        task_names = list(self.optimal_by_task.keys())

        # Check minimal requirements
        if len(task_names) < 2:
            raise ValueError(
                f"Rewind similarity sample requires at least 2 tasks, but only {len(task_names)} available"
            )

        # Select reference task
        task_ref = random.choice(task_names)
        ref_task_indices = self.optimal_by_task[task_ref]

        # Check minimal requirements for reference task
        if len(ref_task_indices) < 2:
            raise ValueError(
                f"Task '{task_ref}' has only {len(ref_task_indices)} trajectory(ies). "
                f"Need at least 2 trajectories in reference task for rewind similarity sample."
            )

        # Select reference trajectory
        ref_idx = random.choice(ref_task_indices)
        ref_traj = self.dataset[ref_idx]

        # traj_sim is a rewound trajectory from same task (different from ref)
        available_sim_indices = [idx for idx in ref_task_indices if idx != ref_idx]
        if not available_sim_indices:
            raise ValueError(
                f"Cannot create rewound traj_sim: no trajectories available in task '{task_ref}' "
                f"different from reference trajectory {ref_traj['id']}"
            )
        sim_idx = random.choice(available_sim_indices)
        traj_sim = self.dataset[sim_idx]

        # traj_diff MUST be from different task
        other_task = random.choice([t for t in task_names if t != task_ref])
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
        diff_idx = random.choice(other_task_indices)
        traj_diff = self.dataset[diff_idx]

        # Final validation
        self._validate_similarity_trajectories(ref_traj, traj_sim, traj_diff)

        # Deserialize frames and create sample
        return self._build_similarity_sample(
            ref_traj, traj_sim, traj_diff, is_rewind=True, strategy_used="rewind_same_task"
        )

    def _create_optimal_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using optimal/suboptimal logic.

        Rules:
        - traj_sim is always from the same task as o^ref (different trajectory)
        - traj_sim is always preferred over traj_diff relative to o^ref
        - If traj_sim is suboptimal, then traj_diff must be from a different task
        - If traj_sim is optimal, then traj_diff can be suboptimal from the same task or from a different task

        Minimal requirements:
        - At least 2 trajectories in the reference task (for o^ref and traj_sim)
        - At least 1 other trajectory available for traj_diff (same or different task)
        """

        # Get available tasks
        task_names = list(self.optimal_by_task.keys())

        # Select reference task
        task_ref = random.choice(task_names)

        # Get optimal and all trajectories from reference task
        ref_optimal_indices = self.optimal_by_task[task_ref]
        ref_all_indices = self.optimal_by_task[task_ref] + self.suboptimal_by_task.get(task_ref, [])

        # Check minimal requirements for reference task
        if len(ref_all_indices) < 2:
            raise ValueError(
                f"Task '{task_ref}' has only {len(ref_all_indices)} trajectory(ies). "
                f"Need at least 2 trajectories in reference task for optimal similarity sample."
            )

        if not ref_optimal_indices:
            # Fall back to all trajectories if no optimal ones
            ref_optimal_indices = ref_all_indices

        # Select reference trajectory (can be optimal or suboptimal)
        ref_idx = random.choice(ref_all_indices)
        ref_traj = self.dataset[ref_idx]

        # Decide if traj_sim should be optimal or suboptimal
        use_optimal_sim = random.choice([True, False])

        if use_optimal_sim and ref_optimal_indices:
            traj_sim, traj_diff = self._select_optimal_sim_trajectories(
                task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices
            )
        else:
            traj_sim, traj_diff = self._select_suboptimal_sim_trajectories(
                task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices
            )

        # Final validation
        self._validate_similarity_trajectories(ref_traj, traj_sim, traj_diff)

        # Deserialize frames and create sample
        return self._build_similarity_sample(
            ref_traj, traj_sim, traj_diff, is_rewind=False, strategy_used="optimal_same_task"
        )

    def _select_optimal_sim_trajectories(self, task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices):
        """Select trajectories when traj_sim should be optimal."""
        # traj_sim is optimal from the same task as ref (must be different trajectory)
        available_sim_indices = [idx for idx in ref_optimal_indices if self.dataset[idx]["id"] != ref_traj["id"]]
        if not available_sim_indices:
            raise ValueError(
                f"Cannot create optimal traj_sim: no optimal trajectories available in task '{task_ref}' "
                f"different from reference trajectory {ref_traj['id']}"
            )
        sim_idx = random.choice(available_sim_indices)
        traj_sim = self.dataset[sim_idx]

        # traj_diff can be suboptimal from same task OR from different task
        if len(task_names) > 1 and random.choice([True, False]):
            # Choose traj_diff from different task
            other_task = random.choice([t for t in task_names if t != task_ref])
            other_task_indices = self.optimal_by_task[other_task]
            if not other_task_indices:
                raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
            diff_idx = random.choice(other_task_indices)
            traj_diff = self.dataset[diff_idx]
        else:
            # Choose traj_diff as suboptimal from same task
            ref_suboptimal_indices = [
                idx
                for idx in ref_all_indices
                if idx not in ref_optimal_indices and self.dataset[idx]["id"] not in [ref_traj["id"], traj_sim["id"]]
            ]
            if not ref_suboptimal_indices:
                # Try any trajectory from same task that's different from ref and traj_sim
                available_same_task_indices = [
                    idx for idx in ref_all_indices if self.dataset[idx]["id"] not in [ref_traj["id"], traj_sim["id"]]
                ]
                if not available_same_task_indices:
                    # Must use different task
                    if len(task_names) < 2:
                        raise ValueError(
                            f"Cannot create traj_diff: only one task available and no trajectories "
                            f"in task '{task_ref}' different from ref {ref_traj['id']} and traj_sim {traj_sim['id']}"
                        )
                    other_task = random.choice([t for t in task_names if t != task_ref])
                    other_task_indices = self.optimal_by_task[other_task]
                    if not other_task_indices:
                        raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
                    diff_idx = random.choice(other_task_indices)
                    traj_diff = self.dataset[diff_idx]
                else:
                    diff_idx = random.choice(available_same_task_indices)
                    traj_diff = self.dataset[diff_idx]
            else:
                diff_idx = random.choice(ref_suboptimal_indices)
                traj_diff = self.dataset[diff_idx]

        return traj_sim, traj_diff

    def _select_suboptimal_sim_trajectories(self, task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices):
        """Select trajectories when traj_sim should be suboptimal."""
        # traj_sim is suboptimal from the same task as ref (must be different trajectory)
        ref_suboptimal_indices = [
            idx
            for idx in ref_all_indices
            if idx not in ref_optimal_indices and self.dataset[idx]["id"] != ref_traj["id"]
        ]
        if not ref_suboptimal_indices:
            # Fallback to any trajectory from same task different from ref
            available_same_task_indices = [idx for idx in ref_all_indices if self.dataset[idx]["id"] != ref_traj["id"]]
            if not available_same_task_indices:
                raise ValueError(
                    f"Cannot create traj_sim: no trajectories available in task '{task_ref}' "
                    f"different from reference trajectory {ref_traj['id']}"
                )
            sim_idx = random.choice(available_same_task_indices)
            traj_sim = self.dataset[sim_idx]
        else:
            sim_idx = random.choice(ref_suboptimal_indices)
            traj_sim = self.dataset[sim_idx]

        # traj_diff MUST be from different task (since traj_sim is suboptimal)
        if len(task_names) < 2:
            raise ValueError(
                f"Cannot create traj_diff: traj_sim is suboptimal so traj_diff must be from different task, "
                f"but only one task '{task_ref}' is available"
            )

        other_task = random.choice([t for t in task_names if t != task_ref])
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
        diff_idx = random.choice(other_task_indices)
        traj_diff = self.dataset[diff_idx]

        return traj_sim, traj_diff

    def _validate_similarity_trajectories(self, ref_traj, traj_sim, traj_diff):
        """Validate that all trajectories have unique IDs."""
        if traj_sim["id"] == ref_traj["id"]:
            raise ValueError(f"traj_sim and o^ref have the same trajectory ID: {traj_sim['id']}")

        if traj_diff["id"] == ref_traj["id"]:
            raise ValueError(f"traj_diff and o^ref have the same trajectory ID: {traj_diff['id']}")

        if traj_sim["id"] == traj_diff["id"]:
            raise ValueError(f"traj_sim and traj_diff have the same trajectory ID: {traj_sim['id']}")

    def _build_similarity_sample(self, ref_traj, traj_sim, traj_diff, is_rewind=False, strategy_used=None):
        """Build the final similarity sample from trajectories."""
        # Get frames from npz files and apply uniform subsampling
        ref_frames_full = self._load_frames_from_npz(ref_traj["frames"])
        traj_sim_frames_full = self._load_frames_from_npz(traj_sim["frames"])
        traj_diff_frames_full = self._load_frames_from_npz(traj_diff["frames"])

        # Uniformly subsample all trajectories to num_frames (default 8)
        num_frames_to_sample = getattr(self.config, 'max_frames', 8)
        ref_frames, ref_indices = self._uniformly_subsample_frames(ref_frames_full, num_frames_to_sample)
        traj_sim_frames, traj_sim_indices = self._uniformly_subsample_frames(traj_sim_frames_full, num_frames_to_sample)
        traj_diff_frames, traj_diff_indices = self._uniformly_subsample_frames(traj_diff_frames_full, num_frames_to_sample)

        # Calculate progress for each trajectory relative to their original frame count
        ref_progress = [idx / (len(ref_frames_full) - 1) for idx in ref_indices]
        traj_sim_progress = [idx / (len(traj_sim_frames_full) - 1) for idx in traj_sim_indices]
        traj_diff_progress = [idx / (len(traj_diff_frames_full) - 1) for idx in traj_diff_indices]
        
        # Store original frame positions for reference
        ref_original_positions = [idx for idx in ref_indices]
        traj_sim_original_positions = [idx for idx in traj_sim_indices]
        traj_diff_original_positions = [idx for idx in traj_diff_indices]

        # Calculate target progress for all trajectories
        # Use subsampled progress if available, otherwise calculate from frames
        target_progress_A = traj_sim_progress
        target_progress_B = traj_diff_progress
        target_progress_ref = ref_progress

        # Ensure all trajectories have exactly max_frames by padding if needed
        ref_frames_padded, target_progress_ref_padded = self._pad_trajectory_to_max_frames(
            ref_frames, target_progress_ref, num_frames_to_sample
        )
        traj_sim_frames_padded, target_progress_A_padded = self._pad_trajectory_to_max_frames(
            traj_sim_frames, target_progress_A, num_frames_to_sample
        )
        traj_diff_frames_padded, target_progress_B_padded = self._pad_trajectory_to_max_frames(
            traj_diff_frames, target_progress_B, num_frames_to_sample
        )

        # Get frame shapes after padding
        ref_frames_shape = ref_frames_padded.shape
        traj_sim_frames_shape = traj_sim_frames_padded.shape
        traj_diff_frames_shape = traj_diff_frames_padded.shape

        # Create similarity sample structure
        sample = SimilaritySample(
            # Create Trajectory objects for reference, traj_sim, and traj_diff
            reference_trajectory=Trajectory(
                frames=ref_frames_padded,
                frames_shape=ref_frames_shape,
                id=ref_traj["id"],
                task=ref_traj["task"],
                lang_vector=ref_traj["lang_vector"],
                data_source=ref_traj["data_source"],
                quality_label=ref_traj.get("quality_label", "successful"),
                is_robot=ref_traj["is_robot"],
                target_progress=target_progress_ref_padded,
                metadata=ref_traj.get("metadata", {})
            ),
            traj_sim_trajectory=Trajectory(
                frames=traj_sim_frames_padded,
                frames_shape=traj_sim_frames_shape,
                id=traj_sim["id"],
                task=traj_sim["task"],
                lang_vector=traj_sim["lang_vector"],
                data_source=traj_sim["data_source"],
                quality_label=traj_sim.get("quality_label"),
                is_robot=traj_sim["is_robot"],
                target_progress=target_progress_A_padded,
                metadata=traj_sim.get("metadata", {})
            ),
            traj_diff_trajectory=Trajectory(
                frames=traj_diff_frames_padded,
                frames_shape=traj_diff_frames_shape,
                id=traj_diff["id"],
                task=traj_diff["task"],
                lang_vector=traj_diff["lang_vector"],
                data_source=traj_diff["data_source"],
                quality_label=traj_diff.get("quality_label"),
                is_robot=traj_diff["is_robot"],
                target_progress=target_progress_B_padded,
                metadata=traj_diff.get("metadata", {})
            ),
            # Data generation info
            data_gen_strategy=strategy_used,
        )

        return sample


class VQADataGenerator(DataGenerator):
    def __init__(self, config, is_evaluation=False):
        self.progress_ratio = config.progress_ratio
        super().__init__(config, is_evaluation)

    def _create_progress_sample(self) -> ProgressSample:
        """Create a progress sample."""
        # Get a random task and optimal trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))
        optimal_idx = random.choice(self.optimal_by_task[task_name])
        traj = self.dataset[optimal_idx]

        # Choose negative generation strategy using configured ratios
        r = random.random()
        rewind_ratio, subopt_ratio, diff_ratio = 0.2, 0.2, 0.2

        strategy_used = None
        if r < 0.6:
            if r < rewind_ratio:
                strategy_choice = 0
            elif r < rewind_ratio + subopt_ratio:
                strategy_choice = 1
            else:
                strategy_choice = 2

            if strategy_choice == 0:
                # Strategy 1: Use rewind-generated suboptimal trajectory from same task
                traj = self._create_rewind_trajectory(traj)
                strategy_used = "rewind_same_task"
            elif strategy_choice == 1:
                # Strategy 2: Use random suboptimal trajectory from same task
                same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
                same_task_suboptimal = [
                    self.dataset[idx] for idx in same_task_suboptimal_indices if self.dataset[idx]["id"] != traj["id"]
                ]
                if same_task_suboptimal:
                    traj = random.choice(same_task_suboptimal)
                    strategy_used = "suboptimal_same_task"
                else:
                    # Fall back to rewind if no same-task suboptimal trajectories
                    traj = self._create_rewind_trajectory(traj)
                    strategy_used = "rewind_same_task"
            else:
                # Strategy 3: Use trajectory from different task (can be optimal or suboptimal)
                other_tasks = [task for task in self.optimal_by_task.keys() if task != traj["task"]]
                if other_tasks:
                    other_task = random.choice(other_tasks)
                    # Get random index from other task and access dataset directly
                    other_task_indices = self.optimal_by_task[other_task]
                    if other_task_indices:
                        other_idx = random.choice(other_task_indices)
                        other_traj = self.dataset[other_idx]
                        # Check if it's not the same trajectory
                        if other_traj["id"] != traj["id"]:
                            traj = other_traj
                            strategy_used = "different_task"
                        else:
                            # Fall back to rewind if same trajectory
                            traj = self._create_rewind_trajectory(traj)
                            strategy_used = "rewind_same_task"
                    else:
                        # Fall back to rewind if no other trajectories available
                        traj = self._create_rewind_trajectory(traj)
                        strategy_used = "rewind_same_task"
                else:
                    # Fall back to rewind if only one task available
                    traj = self._create_rewind_trajectory(traj)
                    strategy_used = "rewind_same_task"

            # Handle negative trajectory frames - could be from dataset (npz) or rewind-generated (numpy)
            if isinstance(traj, dict) and "frames" in traj:
                if isinstance(traj["frames"], str) and traj["frames"].endswith(".npz"):
                    # Regular trajectory with npz path
                    traj_frames = self._load_frames_from_npz(traj["frames"])
                elif isinstance(traj["frames"], np.ndarray):
                    # Rewind trajectory with numpy array
                    traj_frames = traj["frames"]
                else:
                    raise ValueError(f"Unexpected frames format in negative trajectory: {type(traj['frames'])}")
            else:
                raise ValueError(f"Invalid negative trajectory format: {type(traj)}")

        else:
            # Get frames from npz files and uniformly subsample
            traj_frames_full = self._get_trajectory_frames(optimal_idx)
            
            # Uniformly subsample the trajectory to num_frames (default 8)
            num_frames_to_sample = getattr(self.config, 'max_frames', 8)
            traj_frames, traj_indices = self._uniformly_subsample_frames(traj_frames_full, num_frames_to_sample)
            
            # Calculate progress relative to the original trajectory (64 frames)
            traj_progress = [idx / (len(traj_frames_full) - 1) for idx in traj_indices]
            
            # Store original frame positions for reference
            traj_original_positions = [idx for idx in traj_indices]
            
            # Update traj with subsampled frames and progress
            traj = traj.copy()
            traj["frames"] = traj_frames
            traj["frames_shape"] = traj_frames.shape
            traj["metadata"] = traj.get("metadata", {}).copy()
            traj["metadata"]["subsampled_generated"] = True
            traj["metadata"]["subsampled_progress"] = traj_progress
            traj["metadata"]["num_frames_subsampled"] = num_frames_to_sample
            traj["metadata"]["original_num_frames"] = len(traj_frames_full)
            traj["metadata"]["original_frame_positions"] = traj_original_positions

            # Ensure trajectory has exactly max_frames by padding if needed
            traj_frames_padded, traj_progress_padded = self._pad_trajectory_to_max_frames(
                traj_frames, traj_progress, num_frames_to_sample
            )
            
            # Update traj with padded frames and progress
            traj["frames"] = traj_frames_padded
            traj["frames_shape"] = traj_frames_padded.shape
            traj["metadata"]["subsampled_progress"] = traj_progress_padded

        # Calculate target progress for the trajectory
        # Use subsampled progress if available, otherwise calculate from frames
        if traj.get("metadata", {}).get("subsampled_generated"):
            target_progress = traj["metadata"]["subsampled_progress"]
        else:
            target_progress = self._calculate_target_progress(traj, traj_frames)
        
        # Get frame shapes from the trajectory (already padded if needed)
        traj_frames_shape = traj.get("frames_shape")
        if isinstance(traj_frames_shape, list):
            traj_frames_shape = tuple(traj_frames_shape)
        
        # Create progress sample
        sample = ProgressSample(
            frames=traj["frames"],
            frames_shape=traj_frames_shape,
            task=traj["task"],
            target_progress=target_progress,
            quality_label=traj.get("quality_label"),
            sample_type="progress",
        )

        return sample

def test():
    """Test the BatchCollator with generated samples."""
    from transformers import AutoProcessor

    # Create a mock config for testing
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class MockDataConfig:
        train_datasets: List[str] = None
        train_subsets: List[str] = None
        eval_datasets: List[str] = None
        eval_subsets: List[str] = None
        preference_ratio: float = 1.0
        similarity_ratio: float = 0.0
        dataset_preference_ratio: float = 0.7
        shuffle: bool = True
        seed: int = 42
        num_proc: int = 4
        max_frames: int = 8  # Use 8 frames for testing the new subsampling logic
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: List[float] = None

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["abraranwar/libero_rfm"],
        train_subsets=["libero256_90"],
        preference_ratio=1.0,
        similarity_ratio=0.0,
        preference_strategy_ratio=[0.8, 0.1, 0.1, 0.0],
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,  # Use 8 frames for testing the new subsampling logic
        force_reprocess=False,
        model_type="default"
    )

    mock_config = MockConfig(data=mock_data_config, debug=False)

    # Create data generator with mock config
    generator = DataGenerator(config=mock_data_config)

    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")
    from rfm.data.dataset import InfiniteDataGeneratorDataset

    infinite_dataset = InfiniteDataGeneratorDataset(generator)

    preference_count = 0
    similarity_count = 0

    for i in range(10):
        sample = infinite_dataset[i]
        if sample.sample_type == "preference":
            preference_count += 1
        else:
            similarity_count += 1
        rank_0_print(f"Sample {i}: {sample.sample_type}")

    rank_0_print(f"Generated {preference_count} preference samples and {similarity_count} similarity samples")
    rank_0_print(
        f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.similarity_ratio:.1f} similarity"
    )

    # Test the batch collator with infinite dataset
    rank_0_print("\nTesting batch collator with infinite dataset...")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    batch_collator = BatchCollator(processor, max_length=1024)

    # Generate a batch from the infinite dataset
    batch = []
    for i in range(10):  # Generate 4 samples
        sample = infinite_dataset[i]
        batch.append(sample)

    processed_batch = batch_collator(batch)
    for key, value in processed_batch.items():
        rank_0_print(key)
        if key == "preference_inputs":
            for key2, value2 in value.items():
                if key2 != "sample_type":
                    rank_0_print(f"{key2} {value2.shape if hasattr(value2, 'shape') else type(value2)}")
        elif key == "similarity_inputs":
            for key2, value2 in value.items():
                if key2 != "sample_type":
                    rank_0_print(f"{key2} {value2.shape if hasattr(value2, 'shape') else type(value2)}")

    # Do a quick forward pass on RFMModel
    from transformers import Qwen2_5_VLModel, AutoProcessor
    from rfm.models.rfm import RFMModel

    # Load base model and create RFMModel
    base_model = Qwen2_5_VLModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Add RFM special tokens if they don't exist
    special_tokens = ["<|split_token|>", "<|reward_token|>", "<|pref_token|>"]
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            rank_0_print(f"Added special token: {token}")

    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(processor.tokenizer))
        rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")

    rfm_model = RFMModel(config=base_model.config, processor=processor)
    rfm_model.model.load_state_dict(base_model.state_dict())

    # Check if CUDA is available
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    rfm_model = rfm_model.to(device)
    inputs = processed_batch["preference_inputs"]

    # Debug video grid dimensions in test
    rank_0_print(
        f"TEST DEBUG: video_grid_thw shape: {inputs.get('video_grid_thw').shape if inputs.get('video_grid_thw') is not None else None}"
    )
    rank_0_print(
        f"TEST DEBUG: pixel_values_videos shape: {inputs.get('pixel_values_videos').shape if inputs.get('pixel_values_videos') is not None else None}"
    )
    rank_0_print(
        f"TEST DEBUG: second_per_grid_ts shape: {inputs.get('second_per_grid_ts').shape if inputs.get('second_per_grid_ts') is not None else None}"
    )

    outputs = rfm_model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        # pixel_values=inputs.get("pixel_values").to(device),
        pixel_values_videos=inputs.get("pixel_values_videos").to(device),
        # image_grid_thw=inputs.get("image_grid_thw").to(device),
        video_grid_thw=inputs.get("video_grid_thw").to(device),
        second_per_grid_ts=inputs.get("second_per_grid_ts").to(device),
        sample_type="preference",  # Test preference prediction
    )

    rank_0_print("RFM model output structure:")
    rank_0_print(f"  logits: {outputs.shape if outputs is not None else None}")
    rank_0_print(f"  output type: {type(outputs)}")


def test_subsampling():
    """Test both uniform and linspace subsampling logic."""
    from dataclasses import dataclass
    from typing import List
    import numpy as np

    @dataclass
    class MockDataConfig:
        max_frames: int = 8

    @dataclass
    class MockConfig:
        data: MockDataConfig = None

    # Create mock config
    mock_data_config = MockDataConfig(max_frames=8)
    mock_config = MockConfig(data=mock_data_config)

    # Create data generator
    generator = DataGenerator(config=mock_config)

    # Test with 64 frames (as would be the case after preprocessing)
    test_frames = np.random.rand(64, 224, 224, 3)
    
    print("Testing Uniform Subsampling:")
    # Test uniform subsampling
    subsampled_frames, subsampled_indices = generator._uniformly_subsample_frames(test_frames, 8)
    
    print(f"  Original frames: {test_frames.shape}")
    print(f"  Subsampled frames: {subsampled_frames.shape}")
    print(f"  Subsampled indices: {subsampled_indices}")
    
    # Calculate progress from indices
    subsampled_progress = [idx / (len(test_frames) - 1) for idx in subsampled_indices]
    print(f"  Subsampled progress: {[f'{p:.3f}' for p in subsampled_progress]}")
    
    # Verify that we get exactly 8 frames
    assert len(subsampled_frames) == 8, f"Expected 8 frames, got {len(subsampled_frames)}"
    assert len(subsampled_indices) == 8, f"Expected 8 indices, got {len(subsampled_indices)}"
    
    # Verify that progress starts at 0 and is relative to original frame count
    assert subsampled_progress[0] == 0.0, f"First progress should be 0.0, got {subsampled_progress[0]}"
    assert subsampled_progress[-1] == 1.0, f"Last progress should be 1.0, got {subsampled_progress[-1]}"
    
    print("\nTesting Linspace Subsampling:")
    # Test linspace subsampling
    subsampled_frames_lin, subsampled_indices_lin = generator._linspace_subsample_frames(test_frames, 8)
    
    print(f"  Original frames: {test_frames.shape}")
    print(f"  Subsampled frames: {subsampled_frames_lin.shape}")
    print(f"  Subsampled indices: {subsampled_indices_lin}")
    
    # Calculate progress from indices
    subsampled_progress_lin = [idx / (len(test_frames) - 1) for idx in subsampled_indices_lin]
    print(f"  Subsampled progress: {[f'{p:.3f}' for p in subsampled_progress_lin]}")
    
    # Verify that we get exactly 8 frames
    assert len(subsampled_frames_lin) == 8, f"Expected 8 frames, got {len(subsampled_frames_lin)}"
    assert len(subsampled_indices_lin) == 8, f"Expected 8 indices, got {len(subsampled_indices_lin)}"
    
    # Verify that progress starts at 0 and is relative to original frame count
    assert subsampled_progress_lin[0] == 0.0, f"First progress should be 0.0, got {subsampled_progress_lin[0]}"
    assert subsampled_progress_lin[-1] == 1.0, f"Last progress should be 1.0, got {subsampled_progress_lin[-1]}"
    
    # Verify that linspace gives more predictable indices
    expected_linspace_indices = [0, 9, 18, 27, 36, 45, 54, 63]
    assert subsampled_indices_lin == expected_linspace_indices, f"Expected linspace indices {expected_linspace_indices}, got {subsampled_indices_lin}"
    
    print("✅ Both subsampling tests passed!")


def test_rewind_logic():
    """Test the new rewind trajectory creation logic."""
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class MockDataConfig:
        max_frames: int = 8

    @dataclass
    class MockConfig:
        data: MockDataConfig = None

    # Create mock config
    mock_data_config = MockDataConfig(max_frames=8)
    mock_config = MockConfig(data=mock_data_config)

    # Create data generator
    generator = DataGenerator(config=mock_config)

    # Create a mock trajectory with 64 frames
    mock_trajectory = {
        "frames": "dummy_path.npz",  # This won't be used in the test
        "id": "test_traj",
        "task": "test_task",
        "quality_label": "successful"
    }

    # Mock the _load_frames_from_npz method to return test frames
    original_load_frames = generator._load_frames_from_npz
    generator._load_frames_from_npz = lambda x: np.random.rand(64, 224, 224, 3)

    try:
        # Test rewind trajectory creation
        rewind_traj = generator._create_rewind_trajectory(mock_trajectory)
        
        print(f"Rewind trajectory created successfully!")
        print(f"  ID: {rewind_traj['id']}")
        print(f"  Frames shape: {rewind_traj['frames'].shape}")
        print(f"  Quality label: {rewind_traj['quality_label']}")
        
        # Check metadata
        metadata = rewind_traj.get("metadata", {})
        print(f"  Metadata:")
        print(f"    rewind_generated: {metadata.get('rewind_generated')}")
        print(f"    start_idx: {metadata.get('start_idx')}")
        print(f"    end_idx: {metadata.get('end_idx')}")
        print(f"    rewind_point: {metadata.get('rewind_point')}")
        print(f"    start_progress_in_full: {metadata.get('start_progress_in_full')}")
        print(f"    num_frames_subsampled: {metadata.get('num_frames_subsampled')}")
        print(f"    original_num_frames: {metadata.get('original_num_frames')}")
        
        # Verify the progress calculation
        progress = metadata.get('rewind_progress', [])
        if progress:
            print(f"  Progress values: {[f'{p:.3f}' for p in progress]}")
            print(f"  First progress: {progress[0]:.3f} (should be start_progress_in_full)")
            print(f"  Progress range: {progress[-1] - progress[0]:.3f}")
        
        # Verify frame count
        expected_frames = mock_data_config.max_frames
        actual_frames = rewind_traj['frames'].shape[0]
        assert actual_frames == expected_frames, f"Expected {expected_frames} frames, got {actual_frames}"
        
        print("✅ Rewind logic test passed!")
        
    finally:
        # Restore original method
        generator._load_frames_from_npz = original_load_frames


def test_padding():
    """Test the new padding functionality."""
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class MockDataConfig:
        max_frames: int = 8

    @dataclass
    class MockConfig:
        data: MockDataConfig = None

    # Create mock config
    mock_data_config = MockDataConfig(max_frames=8)
    mock_config = MockConfig(data=mock_data_config)

    # Create data generator
    generator = DataGenerator(config=mock_config)

    # Test with frames that need padding (less than max_frames)
    test_frames = np.random.rand(5, 224, 224, 3)  # Only 5 frames
    test_progress = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5 progress values
    
    print(f"Original frames: {test_frames.shape}")
    print(f"Original progress: {test_progress}")
    
    # Test padding
    padded_frames, padded_progress = generator._pad_trajectory_to_max_frames(
        test_frames, test_progress, 8
    )
    
    print(f"Padded frames: {padded_frames.shape}")
    print(f"Padded progress: {padded_progress}")
    
    # Verify padding worked correctly
    assert padded_frames.shape[0] == 8, f"Expected 8 frames, got {padded_frames.shape[0]}"
    assert len(padded_progress) == 8, f"Expected 8 progress values, got {len(padded_progress)}"
    
    # Verify that the first frame and first progress are repeated
    first_frame = test_frames[0]
    first_progress = test_progress[0]
    
    # Check that frames 5-7 are copies of the first frame
    for i in range(5, 8):
        assert np.array_equal(padded_frames[i], first_frame), f"Frame {i} should be copy of first frame"
        assert padded_progress[i] == first_progress, f"Progress {i} should be copy of first progress"
    
    print("✅ Padding test passed!")


def test_subsampling():
    """Test both uniform and linspace subsampling logic."""
    from dataclasses import dataclass
    from typing import List
    import numpy as np

    @dataclass
    class MockDataConfig:
        max_frames: int = 8

    @dataclass
    class MockConfig:
        data: MockDataConfig = None

    # Create mock config
    mock_data_config = MockDataConfig(max_frames=8)
    mock_config = MockConfig(data=mock_data_config)

    # Create data generator
    generator = DataGenerator(config=mock_config)

    # Test with 64 frames (as would be the case after preprocessing)
    test_frames = np.random.rand(64, 224, 224, 3)
    
    print("Testing Uniform Subsampling:")
    # Test uniform subsampling
    subsampled_frames, subsampled_indices = generator._uniformly_subsample_frames(test_frames, 8)
    
    print(f"  Original frames: {test_frames.shape}")
    print(f"  Subsampled frames: {subsampled_frames.shape}")
    print(f"  Subsampled indices: {subsampled_indices}")
    
    # Calculate progress from indices
    subsampled_progress = [idx / (len(test_frames) - 1) for idx in subsampled_indices]
    print(f"  Subsampled progress: {[f'{p:.3f}' for p in subsampled_progress]}")
    
    # Verify that we get exactly 8 frames
    assert len(subsampled_frames) == 8, f"Expected 8 frames, got {len(subsampled_frames)}"
    assert len(subsampled_indices) == 8, f"Expected 8 indices, got {len(subsampled_indices)}"
    
    # Verify that progress starts at 0 and is relative to original frame count
    assert subsampled_progress[0] == 0.0, f"First progress should be 0.0, got {subsampled_progress[0]}"
    assert subsampled_progress[-1] == 1.0, f"Last progress should be 1.0, got {subsampled_progress[-1]}"
    
    print("\nTesting Linspace Subsampling:")
    # Test linspace subsampling
    subsampled_frames_lin, subsampled_indices_lin = generator._linspace_subsample_frames(test_frames, 8)
    
    print(f"  Original frames: {test_frames.shape}")
    print(f"  Subsampled frames: {subsampled_frames_lin.shape}")
    print(f"  Subsampled indices: {subsampled_indices_lin}")
    
    # Calculate progress from indices
    subsampled_progress_lin = [idx / (len(test_frames) - 1) for idx in subsampled_indices_lin]
    print(f"  Subsampled progress: {[f'{p:.3f}' for p in subsampled_progress_lin]}")
    
    # Verify that we get exactly 8 frames
    assert len(subsampled_frames_lin) == 8, f"Expected 8 frames, got {len(subsampled_frames_lin)}"
    assert len(subsampled_indices_lin) == 8, f"Expected 8 indices, got {len(subsampled_indices_lin)}"
    
    # Verify that progress starts at 0 and is relative to original frame count
    assert subsampled_progress_lin[0] == 0.0, f"First progress should be 0.0, got {subsampled_progress_lin[0]}"
    assert subsampled_progress_lin[-1] == 1.0, f"Last progress should be 1.0, got {subsampled_progress_lin[-1]}"
    
    # Verify that linspace gives more predictable indices
    expected_linspace_indices = [0, 9, 18, 27, 36, 45, 54, 63]
    assert subsampled_indices_lin == expected_linspace_indices, f"Expected linspace indices {expected_linspace_indices}, got {subsampled_indices_lin}"
    
    print("✅ Both subsampling tests passed!")


if __name__ == "__main__":
    # test_padding()  # Uncomment to test padding functionality
    # test_subsampling()  # Uncomment to test both uniform and linspace subsampling logic
    # test_rewind_logic()  # Uncomment to test rewind logic
    test()
