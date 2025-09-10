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
from rfm.data.generators.helpers import load_frames_from_npz


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

        return load_frames_from_npz(npz_filepath)
