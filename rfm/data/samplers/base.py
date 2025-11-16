#!/usr/bin/env python3
import numpy as np
import random

from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    subsample_segment_frames,
    compute_progress_from_segment,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
    subsample_pairs_and_progress,
    linspace_subsample_with_cutoff,
    compute_progress_from_linspace,
)
from rfm.data.dataset_types import Trajectory
from rfm.data.datasets.helpers import create_rewind_trajectory, load_embeddings_from_path
from rfm.data.datasets.base import PAIRED_DATA_SOURCES
from rfm.utils.distributed import rank_0_print


class RFMBaseSampler:
    """Base sampler class that provides trajectory retrieval functions for generating samples."""

    def __init__(self, config, dataset, combined_indices, dataset_success_cutoff_map=None, verbose=True):
        """Initialize sampler with dataset and indices.

        Args:
            config: Configuration object
            dataset: The loaded dataset
            combined_indices: Dictionary of combined indices from dataset loading
            dataset_success_cutoff_map: Dictionary mapping dataset names to success cutoff percentages
            verbose: Verbose flag
        """
        self.config = config
        self.dataset = dataset
        self.verbose = verbose
        self.dataset_success_cutoff_map = dataset_success_cutoff_map or {}

        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        # Build indices from combined_indices
        self._build_indices(combined_indices)

        if verbose:
            available_tasks = list(self.task_indices.keys())
            rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}", verbose=self.verbose)
            rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}", verbose=self.verbose)
            rank_0_print(f"  Tasks: {len(self.task_indices)}", verbose=self.verbose)
            rank_0_print(f"  Quality labels: {len(self.quality_indices)}", verbose=self.verbose)
            rank_0_print(f"  Data sources: {len(self.source_indices)}", verbose=self.verbose)
            rank_0_print(f"  Tasks available: {available_tasks[:50]} ...", verbose=self.verbose)
            rank_0_print(f"  Quality labels available: {self.quality_indices.keys()}", verbose=self.verbose)
            rank_0_print(f"  Data sources available: {self.source_indices.keys()}", verbose=self.verbose)

    def _build_indices(self, combined_indices):
        """Build all index mappings from combined_indices.

        Args:
            combined_indices: Dictionary of combined indices from dataset loading
        """
        # Initialize index mappings from the loaded indices
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]
        self.partial_success_indices = combined_indices["partial_success_indices"]

        # Build paired_human_robot_by_task from task_indices after concatenation
        self._build_paired_human_robot_index()

    def _generate_sample(self, item):
        """Generate a sample from an item.

        This method should be overridden by subclasses to implement their specific
        sample generation logic.

        Args:
            item: An item from the dataset (typically a trajectory dict)

        Returns:
            A sample object (e.g., PreferenceSample, SimilaritySample, ProgressSample)
        """
        raise NotImplementedError("Subclasses must implement _generate_sample")

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
            tasks_with_pairs = [
                task
                for task, task_dict in self.paired_human_robot_by_task.items()
                if task_dict["robot"] and task_dict["human"]
            ]
            num_tasks_with_pairs = len(tasks_with_pairs)
            rank_0_print(
                f"  Built paired_human_robot_by_task index: {num_tasks_with_pairs} tasks with both robot and human trajectories (from paired data sources only)",
                verbose=self.verbose,
            )

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

        # Sample a paired trajectory and verify it's different from reference
        chosen_id = ref_traj["id"]
        available_indices = opposite_indices.copy()
        paired_traj = None

        while paired_traj is None or paired_traj.get("id") == chosen_id:
            if not available_indices:
                return None

            paired_idx = random.choice(available_indices)
            paired_traj = self.dataset[paired_idx]

            # If it matches, remove this index and try again
            if paired_traj.get("id") == chosen_id:
                available_indices = [idx for idx in available_indices if idx != paired_idx]
                paired_traj = None
                continue

        return paired_traj

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

    def _get_traj_from_data(self, traj: dict | Trajectory, subsample_strategy: str | None = None) -> Trajectory:
        """Load, subsample, and optionally pad trajectory data and create a Trajectory object.

        When pairwise_progress is enabled, padding is skipped as it's not needed.

        Args:
            traj: Trajectory dict or Trajectory object
            subsample_strategy: Optional strategy for subsampling ("successful", "subsequence", or None for default)

        Returns:
            Trajectory object with loaded and subsampled data (padded only if not using pairwise_progress)
        """
        if isinstance(traj, Trajectory):
            return traj

        frames = None
        video_embeddings = None
        text_embedding = None

        # Use pairwise subsampling if pairwise_progress is enabled
        use_pairwise = self.config.pairwise_progress

        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]
            data = video_embeddings
        else:
            if isinstance(traj["frames"], str):
                frames = load_frames_from_npz(traj["frames"])
            else:
                frames = traj["frames"]
            data = frames

        # Get total frames for progress computation
        if hasattr(data, "shape"):
            num_frames_total = data.shape[0]
        else:
            num_frames_total = len(data)

        if use_pairwise:
            # Use pairwise subsampling for pairwise progress
            subsampled, progress, metadata = subsample_pairs_and_progress(
                data,
                self.config.max_frames,
                progress_pred_type=self.config.progress_pred_type,
            )
            frames_shape = subsampled.shape
            if self.config.load_embeddings:
                video_embeddings = subsampled
            else:
                frames = subsampled
        elif subsample_strategy == "successful":
            # Successful strategy: linspace subsample with end_idx between cutoff and total
            ds_key = traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
            
            subsampled, indices, end_idx = linspace_subsample_with_cutoff(
                data,
                self.config.max_frames,
                num_frames_total,
                success_cutoff,
            )
            frames_shape = subsampled.shape
            progress = compute_progress_from_linspace(
                num_frames_total=num_frames_total,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
            )
            metadata = {
                "end_idx": end_idx,
                "subsampled_indices": indices,
                "strategy": "successful",
            }
            if self.config.load_embeddings:
                video_embeddings = subsampled
            else:
                frames = subsampled
        else:
            # Default/subsequence strategy: segment subsampling (same as before)
            # Get success cutoff from pre-loaded map
            ds_key = traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(data, self.config.max_frames)
            frames_shape = subsampled.shape
            progress = compute_progress_from_segment(
                num_frames_total=num_frames_total,
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
                "strategy": subsample_strategy or "subsequence",
            }
            if self.config.load_embeddings:
                video_embeddings = subsampled
            else:
                frames = subsampled

        # Only pad if not using pairwise progress (pairwise progress doesn't need padding)
        if not use_pairwise:
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
