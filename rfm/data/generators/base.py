#!/usr/bin/env python3
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
import shutil
import os
from pathlib import Path
import torch
from datasets import concatenate_datasets, Dataset
from rfm.utils.logging import rank_0_print
import json


class BaseDataGenerator:
    def __init__(self, config, is_evaluation=False, verbose=True):
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

        # Show available datasets for debugging
        if verbose:
            self.show_available_datasets()

        # Load trajectory dataset
        self._load_trajectory_dataset()

        if verbose:
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

    def _pad_trajectory_to_max_frames(
        self, frames: np.ndarray, progress: List[float], max_frames: int
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

    def _linspace_subsample_frames(self, frames: np.ndarray, num_frames: int = 8) -> Tuple[np.ndarray, List[int]]:
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

    def _randomly_subsample_frames(
        self, frames: np.ndarray, num_frames: int = 8, seed: Optional[int] = None
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

        Example:
            If we have 64 frames and want 8 frames:
            - Random indices: [7, 23, 41, 12, 58, 3, 35, 49] (example)
            - Subsampled frames: frames[7], frames[23], frames[41], etc.
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

    def _subsample_frames_and_progress(self, frames: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        # For trajectory, sample start and end indices to create a segment
        # This makes the progress calculation consistent with rewind trajectories
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
        frames, indices = self._randomly_subsample_frames(segment_frames, self.config.max_frames)

        # Map the subsampled indices to the corresponding progress values from the full segment
        # The chosen_indices tell us which frames from the segment we're using
        progress = [segment_progress[idx] for idx in indices]

        # Ensure both trajectories have exactly max_frames by padding if needed
        # Pad by repeating the first frame and first progress value
        frames, progress = self._pad_trajectory_to_max_frames(frames, progress, self.config.max_frames)

        metadata = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "subsampled_indices": indices,
        }
        return frames, progress, metadata

    def __next__(self):
        return self._create_sample()
