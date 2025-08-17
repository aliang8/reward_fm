#!/usr/bin/env python3
"""
Dataset preprocessing script that creates index-based caches for fast trajectory access.
Uses HuggingFace's .map() for efficient processing and saves trajectory indices.
Handles both training and evaluation datasets separately.
"""

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datasets import Dataset, DatasetDict, load_dataset, Video
import numpy as np
from tqdm import tqdm

from rfm.utils.logging import rank_0_print
from evals.eval_utils import load_experiment_config_from_yaml


class BaseDatasetPreprocessor:
    """Base class for dataset preprocessing with common functionality."""

    def __init__(self, config, cache_dir: str, dataset_type: str):
        self.config = config
        self.cache_dir = cache_dir
        self.dataset_type = dataset_type

        # Add attributes for video processing
        self.max_frames = config.data.max_frames
        self.resized_height = config.data.resized_height
        self.resized_width = config.data.resized_width
        self.force_reprocess = config.data.force_reprocess
        self.num_proc = config.data.num_proc

        # Dataset storage - now store individual datasets
        self.datasets: Dict[str, Dataset] = {}  # key: "dataset_path/subset"
        self.dataset_indices: Dict[str, Dict] = {}  # key: "dataset_path/subset", value: index mappings

    def preprocess_datasets(self):
        """Preprocess each dataset/subset pair individually and create index-based caches."""
        rank_0_print(f"\n🔧 Preprocessing {self.dataset_type} datasets...")

        if self.dataset_type == "training":
            datasets = self.config.data.train_datasets
            subsets = self.config.data.train_subsets
        else:
            datasets = self.config.data.eval_datasets
            subsets = self.config.data.eval_subsets

        # Show which datasets are already preprocessed
        self._show_preprocessed_datasets(datasets, subsets)

        # Process each dataset and its associated subsets
        for i, (dataset_path, dataset_subsets) in enumerate(zip(datasets, subsets)):
            rank_0_print(f"\n📚 Processing {self.dataset_type} dataset {i + 1}/{len(datasets)}: {dataset_path}")

            # Handle both single subset (string) and multiple subsets (list)
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]

            for j, subset in enumerate(dataset_subsets):
                rank_0_print(f"  📂 Processing subset {j + 1}/{len(dataset_subsets)}: {subset}")

                # Create individual cache key
                cache_key = f"{dataset_path}/{subset}"
                individual_cache_dir = os.path.join(self.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

                # Check if already processed
                if os.path.exists(individual_cache_dir) and not self.force_reprocess:
                    rank_0_print(f"    ✅ Cache already exists at {individual_cache_dir}, loading...")
                    self._load_individual_cache(individual_cache_dir, cache_key)
                    continue

                # Load and process individual dataset
                try:
                    dataset = self._load_dataset_from_path(dataset_path, subset)

                    # Handle DatasetDict
                    if isinstance(dataset, DatasetDict):
                        if "train" in dataset:
                            dataset = dataset["train"]
                        else:
                            rank_0_print(f"    ⚠️  Warning: No 'train' split found in {dataset_path}/{subset}")
                            continue

                    rank_0_print(f"    📥 Loaded {len(dataset)} trajectories from {dataset_path}/{subset}")

                    # Process this individual dataset
                    processed_dataset, indices = self._process_individual_dataset(
                        dataset, individual_cache_dir, cache_key
                    )

                    # Store processed dataset and indices
                    self.datasets[cache_key] = processed_dataset
                    self.dataset_indices[cache_key] = indices

                    # Save individual cache
                    self._save_individual_cache(individual_cache_dir, processed_dataset, indices, dataset_path, subset)

                    rank_0_print(f"    ✅ Successfully processed and cached {dataset_path}/{subset}")

                except Exception as e:
                    rank_0_print(f"    ❌ Failed to process {dataset_path}/{subset}: {e}")
                    continue

        if not self.datasets:
            if self.dataset_type == "training":
                raise ValueError("No training datasets were successfully processed")
            else:
                rank_0_print("Warning: No evaluation datasets were successfully processed")
                return

        rank_0_print(f"✅ Successfully processed {len(self.datasets)} {self.dataset_type} datasets")

        # Log summary of processed datasets
        total_trajectories = sum(len(dataset) for dataset in self.datasets.values())
        rank_0_print(f"📊 Total trajectories across all datasets: {total_trajectories}")

        for cache_key, dataset in self.datasets.items():
            rank_0_print(f"  📚 {cache_key}: {len(dataset)} trajectories")

        # Show final status summary
        self._show_final_status_summary(datasets, subsets)

    def _process_individual_dataset(self, dataset: Dataset, cache_dir: str, cache_key: str):
        """Process a single dataset and build its index mappings."""
        # Cast the frames_path column to Video feature
        dataset = dataset.cast_column("frames_path", Video(decode=True))

        # Process videos and build indices
        processed_dataset, indices = self._process_dataset_videos_map(dataset, cache_key)

        return processed_dataset, indices

    def _preprocess_videos(self, frames, num_frames: int = None) -> np.ndarray:
        """
        Process video frames from VideoReader objects into numpy arrays with downsampling.

        Args:
            frames: VideoReader object from HuggingFace Video feature
            num_frames: Number of frames to extract (default: uses self.max_frames)

        Returns:
            frames as numpy arrays with shape (max_frames, H, W, C) where T is time dimension
        """
        if num_frames is None:
            num_frames = self.max_frames

        if frames is None:
            return np.array([])

        try:
            # Convert VideoReader to list of frames
            all_frames = list(frames)
            total_frames = len(all_frames)

            if total_frames == 0:
                return np.array([])

            # Extract frame data from VideoReader objects
            frames_list = []
            for frame in all_frames:
                if hasattr(frame, "get") and "data" in frame:
                    # VideoReader frame format
                    frame_data = frame["data"]
                    if hasattr(frame_data, "numpy"):
                        frame_data = frame_data.numpy()
                    frames_list.append(frame_data)
                else:
                    # Direct frame data
                    if hasattr(frame, "numpy"):
                        frame_data = frame.numpy()
                    else:
                        frame_data = frame
                    frames_list.append(frame_data)

            # Stack frames into numpy array
            if frames_list and hasattr(frames_list[0], "shape"):
                frames_array = np.stack(frames_list)
            else:
                frames_array = np.array(frames_list)

            # Ensure we have the correct shape: (T, H, W, C)
            if len(frames_array.shape) != 4:
                raise ValueError(f"Expected 4D array (T, H, W, C), got shape {frames_array.shape}")

            # Convert from CxHxW to HxWxC if needed
            if frames_array.shape[1] == 3:
                frames_array = np.transpose(frames_array, (0, 2, 3, 1))

            # Downsample frames to max_frames
            if total_frames <= num_frames:
                # If video has fewer frames than requested, pad with last frame
                if total_frames < num_frames:
                    last_frame = frames_array[-1]  # Get the last frame
                    padding_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_frames - total_frames, axis=0)
                    frames_array = np.concatenate([frames_array, padding_frames], axis=0)
            else:
                # Uniform sampling across the video
                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                frames_array = frames_array[frame_indices]

            return frames_array

        except Exception as e:
            rank_0_print(f"Error in _preprocess_videos: {e}")
            return np.array([])

    def _process_dataset_videos_map(self, dataset, cache_key: str):
        """
        Process dataset frames using .map() method for efficient on-the-fly processing.
        Also builds index mappings during the same pass to avoid multiple iterations.
        Frames are saved as .npz files and only file paths are stored in the dataset.

        Args:
            dataset: HuggingFace dataset containing trajectories

        Returns:
            Dataset with processed frame paths and metadata
        """
        # Check if frames are already processed (npz file paths)
        sample_item = dataset[0]
        frames_data = sample_item.get("frames")

        if isinstance(frames_data, str) and frames_data.endswith(".npz") and not self.force_reprocess:
            rank_0_print("Frames already processed as npz file paths, skipping processing.")
            return dataset, {}  # Return empty index mappings if already processed
        elif self.force_reprocess and isinstance(frames_data, str) and frames_data.endswith(".npz"):
            rank_0_print("Force reprocessing enabled. Reprocessing frames despite being already processed.")

        rank_0_print("Processing video frames into npz files and building index mappings...")

        # Debug: Check the dataset structure
        sample_item = dataset[0]
        rank_0_print(f"Sample dataset item keys: {list(sample_item.keys())}")
        rank_0_print(f"Sample item structure: {sample_item}")

        # Initialize index mappings
        robot_trajectories = []
        human_trajectories = []
        optimal_by_task = {}
        suboptimal_by_task = {}
        quality_indices = {}
        task_indices = {}
        source_indices = {}

        # Create frames directory for npz files
        frames_dir = os.path.join(self.cache_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        def process_videos_and_build_indices(example, idx):
            """Process frames and build index mappings in a single pass."""
            # Debug: Log what we're processing
            if idx < 5:  # Only log first 5 examples to avoid spam
                rank_0_print(
                    f"Processing example {idx}: {example.get('id', 'unknown')} - {example.get('task', 'unknown')}"
                )

            # Get the video reader object from the Video feature
            frames = example.get("frames_path")
            if frames is None:
                rank_0_print(f"Warning: No frames_path for example {idx}")
                return {"frames": None, "frames_processed": False}

            try:
                # Process video frames using the _preprocess_videos method
                frames_array = self._preprocess_videos(frames, self.config.data.max_frames)

                if frames_array.size == 0:
                    rank_0_print(f"Warning: No frames processed for example {idx}")
                    return {"frames": None, "frames_processed": False}

                # Save frames as npz file
                frames_filename = f"trajectory_{idx:06d}_{example.get('id', 'unknown')}.npz"
                frames_filepath = os.path.join(frames_dir, frames_filename)

                # Save frames with metadata
                np.savez_compressed(
                    frames_filepath,
                    frames=frames_array,
                    shape=frames_array.shape,
                    num_frames=frames_array.shape[0] if len(frames_array.shape) > 0 else 0,
                )

                # Store file path and metadata in dataset (not the actual frames)
                example["frames"] = frames_filepath  # Store path to npz file
                example["frames_shape"] = frames_array.shape
                example["num_frames"] = frames_array.shape[0] if len(frames_array.shape) > 0 else 0
                example["frames_processed"] = True

                # BUILD INDEX MAPPINGS DURING THE SAME PASS
                # Debug: Log the values we're extracting
                if idx < 5:
                    rank_0_print(
                        f"  Example {idx} - is_robot: {example.get('is_robot', True)}, task: {example.get('task', 'unknown')}, quality: {example.get('quality_label', 'successful')}"
                    )

                # Robot/Human trajectories
                if example.get("is_robot", True):
                    robot_trajectories.append(idx)
                else:
                    human_trajectories.append(idx)

                # Quality-based indices
                quality = example.get("quality_label", "successful")
                if quality not in quality_indices:
                    quality_indices[quality] = []
                quality_indices[quality].append(idx)

                # Task-based indices
                task = example.get("task", "unknown")
                if task not in task_indices:
                    task_indices[task] = []
                task_indices[task].append(idx)

                # Source-based indices
                source = example.get("data_source", "unknown")
                if source not in source_indices:
                    source_indices[source] = []
                source_indices[source].append(idx)

                # Optimal/Suboptimal by task
                if task not in optimal_by_task:
                    optimal_by_task[task] = []
                    suboptimal_by_task[task] = []

                if quality in ["successful", "optimal"]:
                    optimal_by_task[task].append(idx)
                elif quality in ["suboptimal", "failed"]:
                    suboptimal_by_task[task].append(idx)

            except Exception as e:
                rank_0_print(f"Warning: Failed to process video frames for example {idx}: {e}")
                example["frames"] = None
                example["frames_processed"] = False

            # Remove the frames_path since we don't need it anymore
            if "frames_path" in example:
                del example["frames_path"]

            return example

        # Apply the mapping function to the dataset
        processed_dataset = dataset.map(
            process_videos_and_build_indices,
            with_indices=True,
            desc="Processing video frames and building indices",
            num_proc=self.config.data.num_proc,
        )

        # Log the built indices
        rank_0_print(f"Built {self.dataset_type} index mappings for {cache_key}:")
        rank_0_print(f"  Robot trajectories: {len(robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(human_trajectories)}")
        rank_0_print(f"  Tasks: {len(task_indices)}")
        rank_0_print(f"  Quality labels: {len(quality_indices)}")
        rank_0_print(f"  Data sources: {len(source_indices)}")

        return processed_dataset, {
            "robot_trajectories": robot_trajectories,
            "human_trajectories": human_trajectories,
            "optimal_by_task": optimal_by_task,
            "suboptimal_by_task": suboptimal_by_task,
            "quality_indices": quality_indices,
            "task_indices": task_indices,
            "source_indices": source_indices,
        }

    def _save_individual_cache(
        self, cache_dir: str, processed_dataset: Dataset, indices: Dict, dataset_path: str, subset: str
    ):
        """Save the processed dataset and index mappings for an individual dataset/subset."""
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Save the processed dataset
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        processed_dataset.save_to_disk(dataset_cache_dir)

        # Save index mappings
        index_mappings = indices

        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        with open(mappings_file, "w") as f:
            json.dump(index_mappings, f, indent=2)

        # Save dataset info
        dataset_info = {
            "dataset_path": dataset_path,
            "subset": subset,
            "total_trajectories": len(processed_dataset),
            "cache_timestamp": str(datetime.datetime.now()),
            "config_hash": self._get_config_hash(),
        }

        info_file = os.path.join(cache_dir, "dataset_info.json")
        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        rank_0_print(f"Individual {self.dataset_type} cache saved to {cache_dir}")

    def _get_config_hash(self) -> str:
        """Generate a hash of the relevant config parameters."""
        import hashlib

        if self.dataset_type == "training":
            # Create a string representation of relevant training config parameters
            config_str = f"train_{self.config.data.train_datasets}_{self.config.data.train_subsets}_{self.config.data.max_frames}_{self.config.data.resized_height}_{self.config.data.resized_width}"
        else:
            # Create a string representation of relevant evaluation config parameters
            config_str = f"eval_{self.config.data.eval_datasets}_{self.config.data.eval_subsets}_{self.config.data.max_frames}_{self.config.data.resized_height}_{self.config.data.resized_width}"

        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_individual_cache(self, cache_dir: str, cache_key: str):
        """Load a pre-processed dataset and its index mappings from a cache directory."""
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        if not os.path.exists(dataset_cache_dir):
            raise FileNotFoundError(f"Processed dataset not found at {dataset_cache_dir}")

        # Load the processed dataset
        self.datasets[cache_key] = Dataset.load_from_disk(dataset_cache_dir)

        # Load index mappings
        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        if not os.path.exists(mappings_file):
            raise FileNotFoundError(f"Index mappings not found at {mappings_file}")

        with open(mappings_file, "r") as f:
            self.dataset_indices[cache_key] = json.load(f)

        # Log loaded cache
        rank_0_print(f"  📂 Loaded individual {self.dataset_type} cache from {cache_dir}")

    def get_combined_indices(self):
        """Get combined index mappings from all individual datasets."""
        if not self.dataset_indices:
            return {}

        # Combine indices from all datasets
        combined_indices = {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": {},
            "suboptimal_by_task": {},
            "quality_indices": {},
            "task_indices": {},
            "source_indices": {},
        }

        # Track offset for each dataset
        offset = 0

        for cache_key, indices in self.dataset_indices.items():
            # Adjust indices by adding offset
            for key in combined_indices:
                if key in indices:
                    if isinstance(indices[key], list):
                        # For list indices, add offset
                        combined_indices[key].extend([idx + offset for idx in indices[key]])
                    elif isinstance(indices[key], dict):
                        # For dict indices, add offset to values
                        if key not in combined_indices[key]:
                            combined_indices[key] = {}
                        for subkey, subindices in indices[key].items():
                            if subkey not in combined_indices[key]:
                                combined_indices[key][subkey] = []
                            combined_indices[key][subkey].extend([idx + offset for idx in subindices])

            # Update offset for next dataset
            if cache_key in self.datasets:
                offset += len(self.datasets[cache_key])

        return combined_indices

    def _load_dataset_from_path(self, dataset_path: str, subset: str = None):
        """Load dataset from path with proper video handling."""
        if "/" in dataset_path and not os.path.exists(dataset_path):
            # Loading from HuggingFace Hub - handle video paths
            rank_0_print(f"Loading from HuggingFace Hub: {dataset_path}")

            # Check if RFM_DATASET_PATH is set
            rfm_dataset_path = os.environ.get("RFM_DATASET_PATH")
            if not rfm_dataset_path:
                raise ValueError(
                    "RFM_DATASET_PATH environment variable not set. "
                    "Please set it to the directory containing your downloaded datasets. "
                    "Example: export RFM_DATASET_PATH=/path/to/your/datasets"
                )

            dataset_name = dataset_path.split("/")[-1]

            def patch_path(old_path):
                # RFM_DATASET_PATH is set in the environment variable
                root_dir = f"{rfm_dataset_path}/{dataset_name}"
                return f"{root_dir}/{old_path}"  # e.g., "./videos/trajectory_0000.mp4"

            # Load dataset with subset
            if subset:
                dataset = load_dataset(dataset_path, name=subset, split="train")
            else:
                dataset = load_dataset(dataset_path, split="train")

            # Just patch the paths, don't decode videos yet
            dataset = dataset.map(lambda x: {"frames_path": patch_path(x["frames"])})
            return dataset
        else:
            # Load from local disk
            if subset:
                dataset = load_dataset(dataset_path, subset)
            else:
                dataset = load_dataset(dataset_path)
            return dataset

    def _show_preprocessed_datasets(self, datasets: List[str], subsets: List[str]):
        """
        Show which datasets are already preprocessed and which are not.
        This helps avoid re-processing already cached datasets.
        """
        rank_0_print(f"\n🔍 Checking for pre-existing {self.dataset_type} caches...")

        cached_count = 0
        total_count = 0

        for i, (dataset_path, dataset_subsets) in enumerate(zip(datasets, subsets)):
            # Handle both single subset (string) and multiple subsets (list)
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]

            total_count += len(dataset_subsets)

            for subset in dataset_subsets:
                cache_key = f"{dataset_path}/{subset}"
                individual_cache_dir = os.path.join(self.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

                if os.path.exists(individual_cache_dir):
                    cached_count += 1
                    # Try to load cache info to show details
                    info_file = os.path.join(individual_cache_dir, "dataset_info.json")
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, "r") as f:
                                info = json.load(f)
                            trajectories = info.get("total_trajectories", "unknown")
                            timestamp = info.get("cache_timestamp", "unknown")
                            rank_0_print(
                                f"  ✅ {dataset_path}/{subset}: {trajectories} trajectories (cached at {timestamp})"
                            )
                        except:
                            rank_0_print(f"  ✅ {dataset_path}/{subset}: Cache exists but info file corrupted")
                    else:
                        rank_0_print(f"  ✅ {dataset_path}/{subset}: Cache exists (no info file)")
                else:
                    rank_0_print(f"  ❌ {dataset_path}/{subset}: No cache found")

        # Show summary
        rank_0_print(f"\n📊 Cache Status Summary:")
        rank_0_print(f"  ✅ Already cached: {cached_count}/{total_count} dataset/subset pairs")
        rank_0_print(f"  🔄 Need processing: {total_count - cached_count}/{total_count} dataset/subset pairs")

        if cached_count == total_count:
            rank_0_print(f"  🎉 All dataset/subset pairs are already cached! Use --force-reprocess to reprocess.")
        elif cached_count > 0:
            rank_0_print(f"  💡 Some dataset/subset pairs are cached. Only uncached ones will be processed.")
        else:
            rank_0_print(f"  🚀 No dataset/subset pairs are cached. All will be processed.")

    def _show_final_status_summary(self, datasets: List[str], subsets: List[str]):
        """
        Show a summary of which datasets were processed and which were loaded from cache.
        """
        rank_0_print(f"\n📊 Final Status Summary for {self.dataset_type} Preprocessing:")

        processed_count = 0
        loaded_count = 0
        total_count = 0

        for i, (dataset_path, dataset_subsets) in enumerate(zip(datasets, subsets)):
            # Handle both single subset (string) and multiple subsets (list)
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]

            total_count += len(dataset_subsets)

            for subset in dataset_subsets:
                cache_key = f"{dataset_path}/{subset}"
                individual_cache_dir = os.path.join(self.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

                if cache_key in self.datasets:
                    if os.path.exists(individual_cache_dir):
                        loaded_count += 1
                        rank_0_print(
                            f"  ✅ {dataset_path}/{subset}: Loaded from cache ({len(self.datasets[cache_key])} trajectories)"
                        )
                    else:
                        processed_count += 1
                        rank_0_print(
                            f"  🔄 {dataset_path}/{subset}: Newly processed ({len(self.datasets[cache_key])} trajectories)"
                        )
                else:
                    rank_0_print(f"  ❌ {dataset_path}/{subset}: Failed to load/process")

        # Show summary counts
        rank_0_print(f"\n📈 Processing Summary:")
        rank_0_print(f"  🔄 Newly processed: {processed_count} dataset/subset pairs")
        rank_0_print(f"  ✅ Loaded from cache: {loaded_count} dataset/subset pairs")
        rank_0_print(f"  ❌ Failed: {total_count - processed_count - loaded_count} dataset/subset pairs")
        rank_0_print(f"  📊 Total available: {processed_count + loaded_count}/{total_count} dataset/subset pairs")


class TrainingDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor specifically for training datasets."""

    def __init__(self, config):
        super().__init__(config, "./processed_datasets/train_cache", "training")


class EvaluationDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor specifically for evaluation datasets."""

    def __init__(self, config):
        super().__init__(config, "./processed_datasets/eval_cache", "evaluation")


def main():
    """Main preprocessing function."""
    # Load config
    config_path = "rfm/configs/config.yaml"  # Adjust path as needed
    config = load_experiment_config_from_yaml(config_path)

    # Show dataset structure info
    print("\n🏗️  Dataset Configuration Structure:")
    print("Training datasets:")
    for dataset_path, dataset_subsets in zip(config.data.train_datasets, config.data.train_subsets):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            print(f"    📂 {subset}")

    print("Evaluation datasets:")
    for dataset_path, dataset_subsets in zip(config.data.eval_datasets, config.data.eval_subsets):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            print(f"    📂 {subset}")

    print("\n💡 Note: Each dataset can now have multiple subsets!")
    print("   - Single subset: ['subset1'] or 'subset1'")
    print("   - Multiple subsets: ['subset1', 'subset2', 'subset3']")

    # Create separate preprocessors for training and evaluation
    train_preprocessor = TrainingDatasetPreprocessor(config)
    eval_preprocessor = EvaluationDatasetPreprocessor(config)

    # Preprocess training datasets
    print("\n=== Processing Training Datasets ===")
    train_preprocessor.preprocess_datasets()

    # Preprocess evaluation datasets
    print("\n=== Processing Evaluation Datasets ===")
    eval_preprocessor.preprocess_datasets()

    # Test the caches
    print("\n=== Testing Caches ===")

    # Training cache
    if train_preprocessor.datasets:
        print(f"\n📚 Training Datasets:")
        total_train_trajectories = sum(len(dataset) for dataset in train_preprocessor.datasets.values())
        print(f"  Total trajectories: {total_train_trajectories}")

        # Get combined indices
        combined_indices = train_preprocessor.get_combined_indices()
        if combined_indices:
            print(f"  Robot trajectories: {len(combined_indices.get('robot_trajectories', []))}")
            print(f"  Human trajectories: {len(combined_indices.get('human_trajectories', []))}")
            print(f"  Tasks: {list(combined_indices.get('task_indices', {}).keys())}")

        # Test direct access to first dataset
        if train_preprocessor.datasets:
            first_dataset = next(iter(train_preprocessor.datasets.values()))
            if len(first_dataset) > 0:
                test_traj = first_dataset[0]
                print(f"  Sample trajectory: {test_traj['id']} - {test_traj['task']}")

    # Evaluation cache
    if eval_preprocessor.datasets:
        print(f"\n📚 Evaluation Datasets:")
        total_eval_trajectories = sum(len(dataset) for dataset in eval_preprocessor.datasets.values())
        print(f"  Total trajectories: {total_eval_trajectories}")

        # Get combined indices
        combined_indices = eval_preprocessor.get_combined_indices()
        if combined_indices:
            print(f"  Robot trajectories: {len(combined_indices.get('robot_trajectories', []))}")
            print(f"  Human trajectories: {len(combined_indices.get('human_trajectories', []))}")
            print(f"  Tasks: {list(combined_indices.get('task_indices', {}).keys())}")

        # Test direct access to first dataset
        if eval_preprocessor.datasets:
            first_dataset = next(iter(eval_preprocessor.datasets.values()))
            if len(first_dataset) > 0:
                test_traj = first_dataset[0]
                print(f"  Sample trajectory: {test_traj['id']} - {test_traj['task']}")

    print("\n✅ Dataset preprocessing complete!")
    print(f"Training cache: {train_preprocessor.cache_dir}")
    print(f"Evaluation cache: {eval_preprocessor.cache_dir}")

    # Show individual dataset info
    print(f"\n📊 Individual Dataset Summary:")
    print(f"Training datasets processed: {len(train_preprocessor.datasets)}")
    for cache_key, dataset in train_preprocessor.datasets.items():
        print(f"  ✅ {cache_key}: {len(dataset)} trajectories")

    print(f"Evaluation datasets processed: {len(eval_preprocessor.datasets)}")
    for cache_key, dataset in eval_preprocessor.datasets.items():
        print(f"  ✅ {cache_key}: {len(dataset)} trajectories")

    # Show dataset structure
    print(f"\n🏗️  Dataset Structure:")
    print(f"Training datasets:")
    for dataset_path, dataset_subsets in zip(
        train_preprocessor.config.data.train_datasets, train_preprocessor.config.data.train_subsets
    ):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            cache_key = f"{dataset_path}/{subset}"
            if cache_key in train_preprocessor.datasets:
                print(f"    ✅ {subset}: {len(train_preprocessor.datasets[cache_key])} trajectories")
            else:
                print(f"    ❌ {subset}: Failed to load")

    print(f"Evaluation datasets:")
    for dataset_path, dataset_subsets in zip(
        eval_preprocessor.config.data.eval_datasets, eval_preprocessor.config.data.eval_subsets
    ):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            cache_key = f"{dataset_path}/{subset}"
            if cache_key in eval_preprocessor.datasets:
                print(f"    ✅ {subset}: {len(eval_preprocessor.datasets[cache_key])} trajectories")
            else:
                print(f"    ❌ {subset}: Failed to load")


if __name__ == "__main__":
    main()
