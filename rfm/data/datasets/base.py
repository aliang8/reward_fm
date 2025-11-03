#!/usr/bin/env python3
import json
import os

import numpy as np
import torch
import random

from datasets import Dataset, concatenate_datasets
from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    load_dataset_success_percent,
    subsample_segment_frames,
    compute_progress_from_segment,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
)
from rfm.utils.distributed import rank_0_print
from rfm.data.dataset_types import Trajectory
from rfm.data.datasets.helpers import create_rewind_trajectory, load_embeddings_from_path

# Global list of data sources that contain paired human/robot trajectories
PAIRED_DATA_SOURCES = [
    "ph2d",
    "motif_rfm",
    "rh20t"
]


class RFMBaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        super().__init__()
        self.config = config
        self.is_evaluation = is_evaluation

        # Choose datasets based on whether this is for evaluation or training
        if is_evaluation and config.eval_datasets:
            self.datasets = config.eval_datasets
        else:
            self.datasets = config.train_datasets

        self.verbose = verbose

        # Initialize dataset and index mappings
        self.dataset = None
        self.robot_trajectories = []
        self.human_trajectories = []
        self.optimal_by_task = {}
        self.suboptimal_by_task = {}
        self.quality_indices = {}
        self.task_indices = {}
        self.source_indices = {}
        self.partial_success_indices = {}
        self.paired_human_robot_by_task = {}

        # Load dataset-specific success cutoff map if available
        self.dataset_success_cutoff_map = {}
        if hasattr(config, "dataset_success_cutoff_file") and config.dataset_success_cutoff_file:
            self.dataset_success_cutoff_map = load_dataset_success_percent(config.dataset_success_cutoff_file)

        # Load trajectory dataset
        self._load_trajectory_dataset()

        # Filter dataset
        # We only want to iterate through successful trajectories
        # self.filtered_dataset = self.dataset.filter(lambda x: x["quality_label"] == "successful")
        self.filtered_dataset = self.dataset

        if verbose:
            available_tasks = list(self.task_indices.keys())

            rank_0_print(f"Dataset initialized with {len(self.dataset)} total trajectories")
            rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}")
            rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}")
            rank_0_print(f"  Tasks: {len(self.task_indices)}")
            rank_0_print(f"  Quality labels: {len(self.quality_indices)}")
            rank_0_print(f"  Data sources: {len(self.source_indices)}")
            rank_0_print(f"  Tasks available: {available_tasks[:50]} ...")
            rank_0_print(f"  Quality labels available: {self.quality_indices.keys()}")
            rank_0_print(f"  Data sources available: {self.source_indices.keys()}")

    def __len__(self):
        return len(self.filtered_dataset)

    def _load_trajectory_dataset(self):
        """Load trajectory dataset using preprocessed index-based cache."""
        cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        if not cache_dir:
            raise ValueError(
                "RFM_PROCESSED_DATASETS_PATH environment variable not set. Please set it to the directory containing your processed datasets."
            )
        cache_type = "evaluation" if self.is_evaluation else "training"

        # Check if preprocessed cache exists
        if os.path.exists(cache_dir):
            rank_0_print(
                f"Found preprocessed cache at {cache_dir}, loading {cache_type} datasets...", verbose=self.verbose
            )

            self._load_preprocessed_cache(cache_dir, is_training=not self.is_evaluation)

            rank_0_print(
                f"Successfully loaded preprocessed {cache_type} datasets with {len(self.dataset)} trajectory indices",
                verbose=self.verbose,
            )
        else:
            # If no cache exists, we need to run the preprocessor first
            rank_0_print("No preprocessed cache found. Please run preprocess_datasets.py first to create the cache.")
            raise RuntimeError(
                "Dataset preprocessing required. Please run:\n"
                "uv run scripts/preprocess_datasets.py\n"
                "This will create the necessary index-based cache for efficient data loading."
            )

    def _load_preprocessed_cache(self, cache_dir: str, is_training: bool = True):
        """Load the preprocessed cache with index mappings for datasets."""
        # Check which datasets are available
        available_datasets = []
        missing_datasets = []

        for dataset_path in self.datasets:
            # The preprocessing script creates individual cache directories for each dataset
            individual_cache_dir = os.path.join(cache_dir, dataset_path.replace("/", "_").replace(":", "_"))

            if os.path.exists(individual_cache_dir):
                info_file = os.path.join(individual_cache_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    try:
                        with open(info_file) as f:
                            json.load(f)

                        available_datasets.append((dataset_path, individual_cache_dir))
                        rank_0_print(f"       Found cache: {individual_cache_dir}", verbose=self.verbose)
                    except:
                        rank_0_print(
                            f"       Cache info file corrupted, skipping: {individual_cache_dir}", verbose=self.verbose
                        )
                        continue
                else:
                    rank_0_print(f"       No info file found, skipping: {individual_cache_dir}", verbose=self.verbose)
                    continue
            else:
                missing_datasets.append(dataset_path)
                rank_0_print(f"      ❌ Missing cache: {individual_cache_dir}", verbose=self.verbose)

        # Warn about missing datasets
        if missing_datasets:
            rank_0_print(
                "\n⚠️  Warning: The following configured datasets are not available in the cache:", verbose=self.verbose
            )
            for dataset_path in missing_datasets:
                rank_0_print(f"    ❌ {dataset_path}", verbose=self.verbose)
            rank_0_print(
                "  Available datasets will be loaded, but some configured data may be missing.", verbose=self.verbose
            )

        if not available_datasets:
            raise RuntimeError(
                f"No configured datasets are available in the cache. "
                f"Please run preprocess_datasets.py to create the cache for: {self.datasets}"
            )

        rank_0_print(
            f"\nSummary: {len(available_datasets)} available, {len(missing_datasets)} missing", verbose=self.verbose
        )

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
            "partial_success_indices": {},
        }

        offset = 0

        for dataset_path, individual_cache_dir in available_datasets:
            # Load the processed dataset
            dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
            if not os.path.exists(dataset_cache_dir):
                rank_0_print(
                    f"   Warning: Processed dataset not found at {dataset_cache_dir}, skipping...", verbose=self.verbose
                )
                continue

            dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=True)
            loaded_datasets.append(dataset)

            # Load index mappings
            mappings_file = os.path.join(individual_cache_dir, "index_mappings.json")
            if os.path.exists(mappings_file):
                with open(mappings_file) as f:
                    indices = json.load(f)

                # Adjust indices by adding offset and combine
                for key in combined_indices:
                    if key in indices:
                        if isinstance(indices[key], list):
                            # For list indices, add offset
                            combined_indices[key].extend([idx + offset for idx in indices[key]])
                        elif isinstance(indices[key], dict):
                            # For regular dict indices, add offset to values
                            for subkey, subindices in indices[key].items():
                                if subkey not in combined_indices[key]:
                                    combined_indices[key][subkey] = []
                                combined_indices[key][subkey].extend([idx + offset for idx in subindices])

            if self.verbose:
                rank_0_print(f"  ✅ Loaded {len(dataset)} trajectories from {dataset_path}", verbose=self.verbose)
            offset += len(dataset)

        if not loaded_datasets:
            raise RuntimeError("No datasets could be loaded from the cache")

        # Concatenate datasets if multiple
        if len(loaded_datasets) == 1:
            self.dataset = loaded_datasets[0]
        else:
            self.dataset = concatenate_datasets(loaded_datasets)

        # Store the combined index mappings
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]
        self.partial_success_indices = combined_indices["partial_success_indices"]
        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]
        
        # Build paired_human_robot_by_task from task_indices after concatenation
        self._build_paired_human_robot_index()

        dataset_type = "training" if is_training else "evaluation"
        rank_0_print(
            f"✅ Loaded {len(self.dataset)} total trajectories from preprocessed {dataset_type} datasets",
            verbose=self.verbose,
        )
        rank_0_print(
            f"  📊 Available datasets: {len(available_datasets)}/{len(missing_datasets) + len(available_datasets)}",
            verbose=self.verbose,
        )
        rank_0_print(f"  📊 Missing datasets: {len(missing_datasets)}", verbose=self.verbose)

    def _build_paired_human_robot_index(self):
        """Build paired_human_robot_by_task index from task_indices by checking is_robot field.
        
        This builds the index after concatenation by iterating through task_indices
        and checking the is_robot field for each trajectory. Only includes trajectories
        from PAIRED data sources.
        """
        self.paired_human_robot_by_task = {}
        
        # Filter indices for paired data sources
        paired_data_source_indices = set()
        for data_source in PAIRED_DATA_SOURCES:
            if data_source in self.source_indices:
                paired_data_source_indices.update(self.source_indices[data_source])
        
        if not paired_data_source_indices:
            if self.verbose:
                rank_0_print("  No paired data sources found, skipping paired index building", verbose=self.verbose)
            return
        
        # Build index from task_indices using cached is_robot field, but only for paired data sources
        for task, indices in self.task_indices.items():
            # Filter to only paired data sources
            task_indices_paired = [idx for idx in indices if idx in paired_data_source_indices]
            
            if not task_indices_paired:
                continue
            
            self.paired_human_robot_by_task[task] = {"robot": [], "human": []}
            
            for idx in task_indices_paired:
                is_robot = self._cached_is_robot[idx] if idx < len(self._cached_is_robot) else True
                if is_robot:
                    self.paired_human_robot_by_task[task]["robot"].append(idx)
                else:
                    self.paired_human_robot_by_task[task]["human"].append(idx)
        
        if self.verbose:
            # Count tasks with both robot and human trajectories
            tasks_with_pairs = [task for task, task_dict in self.paired_human_robot_by_task.items() if task_dict["robot"] and task_dict["human"]]
            num_tasks_with_pairs = len(tasks_with_pairs)
            rank_0_print(f"  Built paired_human_robot_by_task index: {num_tasks_with_pairs} tasks with both robot and human trajectories (from paired data sources only)", verbose=self.verbose)

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

    def _get_same_task_optimal(self, ref_traj: dict) -> dict | None:
        """Get optimal trajectory from same task (different from ref).

        Args:
            ref_traj: Reference trajectory

        Returns:
            Same task optimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_optimal_indices = self.optimal_by_task.get(task_name, [])
        if not same_task_optimal_indices:
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_optimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_optimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_optimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_optimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                return None

        return self.dataset[random_idx]

    def _get_same_task_suboptimal(self, ref_traj: dict) -> dict | None:
        """Get suboptimal trajectory from same task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Suboptimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if not same_task_suboptimal_indices:
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_suboptimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_suboptimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_suboptimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_suboptimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                return None

        return self.dataset[random_idx]

    def _get_different_task(self, ref_traj: dict) -> dict | None:
        """Get trajectory from different task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Different task trajectory dict or None if not available
        """
        other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]
        if not other_tasks:
            return None

        other_task = random.choice(other_tasks)
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            return None

        other_idx = random.choice(other_task_indices)
        return self.dataset[other_idx]

    def _get_paired_human_robot_traj(self, ref_traj: dict) -> dict | None:
        """Get paired human/robot trajectory for the same task.

        Given a reference trajectory, if it's a robot trajectory, returns a human trajectory
        from the same task. If it's a human trajectory, returns a robot trajectory from the
        same task.

        Args:
            ref_traj: Reference trajectory (can be robot or human)

        Returns:
            Paired trajectory dict (opposite type) or None if not available
        """
        task = ref_traj["task"]
        is_robot = ref_traj.get("is_robot", True)
        
        if task not in self.paired_human_robot_by_task:
            return None
        
        task_pairs = self.paired_human_robot_by_task[task]
        
        # Get opposite type
        opposite_key = "human" if is_robot else "robot"
        opposite_indices = task_pairs.get(opposite_key, [])
        
        if not opposite_indices:
            return None
        
        # Exclude the current trajectory if it's in the list
        chosen_id = ref_traj["id"]
        filtered_indices = [
            idx for idx in opposite_indices 
            if self._cached_ids[idx] != chosen_id
        ]
        
        if not filtered_indices:
            # If no other trajectories available, use any from the list
            filtered_indices = opposite_indices
        
        paired_idx = random.choice(filtered_indices)
        return self.dataset[paired_idx]

    def _get_rewound_traj(self, ref_traj: dict) -> Trajectory:
        """Get rewound trajectory from reference trajectory.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Rewound trajectory as Trajectory object (already processed)
        """
        ds_key = ref_traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
        return create_rewind_trajectory(
            ref_traj,
            max_frames=self.config.max_frames,
            use_embeddings=self.config.load_embeddings,
            progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
            success_cutoff=success_cutoff,
        )

    def _get_traj_from_data(self, traj: dict | Trajectory) -> Trajectory:
        """Load, subsample, pad trajectory data and create a Trajectory object.

        Args:
            traj: Trajectory dict or Trajectory object

        Returns:
            Trajectory object with loaded, subsampled, and padded data
        """
        if isinstance(traj, Trajectory):
            return traj

        frames = None
        video_embeddings = None
        text_embedding = None

        if self.config.load_embeddings and traj.get("embeddings_path"):
            video_embeddings = load_embeddings_from_path(traj["embeddings_path"], "video_embeddings")
            text_embedding = load_embeddings_from_path(traj["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(video_embeddings, self.config.max_frames)
            frames_shape = subsampled.shape
            progress = compute_progress_from_segment(
                num_frames_total=self.config.max_frames_after_preprocessing,
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            video_embeddings = subsampled
        else:
            if isinstance(traj["frames"], str):
                frames = load_frames_from_npz(traj["frames"])
            else:
                frames = traj["frames"]

            # Get success cutoff from pre-loaded map
            ds_key = traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(frames, self.config.max_frames)
            frames_shape = subsampled.shape
            progress = compute_progress_from_segment(
                num_frames_total=self.config.max_frames_after_preprocessing,
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            frames = subsampled

        if self.config.load_embeddings:
            video_embeddings, progress = pad_trajectory_to_max_frames_torch(
                video_embeddings, progress, self.config.max_frames
            )
        else:
            frames, progress = pad_trajectory_to_max_frames_np(frames, progress, self.config.max_frames)

        return Trajectory(
            frames=frames,
            frames_shape=frames_shape,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            id=traj["id"],
            task=traj["task"],
            lang_vector=traj["lang_vector"],
            data_source=traj["data_source"],
            quality_label=traj.get("quality_label"),
            is_robot=traj["is_robot"],
            target_progress=progress,
            metadata=metadata,
        )
