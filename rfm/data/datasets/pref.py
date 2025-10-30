#!/usr/bin/env python3
"""
PrefDataset class for producing batches of preference prediction data.
"""

import random

from rfm.data.dataset_types import PreferenceSample, Trajectory
from .base import RFMBaseDataset
from .helpers import (
    DataGenStrat,
    create_rewind_trajectory,
    load_frames_from_npz,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    subsample_segment_frames,
    compute_progress_from_segment,
)
from rfm.utils.distributed import rank_0_print
from rfm.utils.timer import timer
from .helpers import load_embeddings_from_path


class PrefDataset(RFMBaseDataset):
    """Data generator for producing batches of preference prediction data."""

    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        super().__init__(config, is_evaluation, verbose=verbose, **kwargs)

        self.dataset_preference_ratio = config.dataset_preference_ratio
        self.preference_strategy_ratio: list[float] = config.preference_strategy_ratio

        # Initialize preference dataset
        self._load_preference_dataset()

        rank_0_print(f"PrefDataset initialized with {len(self.dataset)} total trajectories")

    def __getitem__(self, idx):
        """Iterate over one sample per trajectory in the dataset."""
        dataset_len = len(self.dataset)
        chosen_traj = self.dataset[idx % dataset_len]
        sample = self._create_pref_sample(chosen_traj)
        return sample

    def _create_video_binned_trajectory(self, original_traj: dict, num_bins: int = 10) -> tuple[dict, dict]:
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
        frames_data = load_frames_from_npz(original_traj["frames"])

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

        chosen_progress = []
        for i in range(len(chosen_frames)):
            chosen_progress.append((i + 1) / (len(frames_data) - chosen_start))

        # Extract frames from the rejected bin (this will be the "rejected" trajectory)
        rejected_start, rejected_end = bin_boundaries[rejected_bin_idx]
        rejected_frames = frames_data[rejected_start:rejected_end]

        rejected_progress = []
        for i in range(len(rejected_frames)):
            rejected_progress.append((i + 1) / (len(frames_data) - rejected_start))

        # Apply uniform subsampling to both bins to ensure consistent frame counts
        # Use uniform subsampling for real trajectories (not rewound)
        num_frames_to_sample = self.config.max_frames
        chosen_frames, chosen_indices = self._linspace_subsample_frames(chosen_frames, num_frames_to_sample)
        rejected_frames, rejected_indices = self._linspace_subsample_frames(rejected_frames, num_frames_to_sample)

        # Calculate progress for each bin relative to the original trajectory
        chosen_progress = [chosen_progress[idx] for idx in chosen_indices]
        rejected_progress = [rejected_progress[idx] for idx in rejected_indices]

        # Create the chosen trajectory (from chosen bin)
        chosen_traj = original_traj.copy()
        chosen_traj["frames"] = chosen_frames
        chosen_traj["frames_shape"] = chosen_frames.shape
        chosen_traj["target_progress"] = chosen_progress
        chosen_traj["metadata"] = {
            "start_idx": chosen_start,
            "end_idx": chosen_end,
            "chosen_bin_idx": chosen_bin_idx,
            "rejected_bin_idx": rejected_bin_idx,
        }

        rejected_traj = original_traj.copy()
        rejected_traj["frames"] = rejected_frames
        rejected_traj["frames_shape"] = rejected_frames.shape
        rejected_traj["target_progress"] = rejected_progress
        rejected_traj["metadata"] = {
            "start_idx": rejected_start,
            "end_idx": rejected_end,
            "chosen_bin_idx": chosen_bin_idx,
            "rejected_bin_idx": rejected_bin_idx,
        }

        return chosen_traj, rejected_traj

    def _create_pref_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            raise ValueError("No preferences loaded from dataset")

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        raise NotImplementedError("Preference sample creation from dataset not yet implemented")

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_print("No preference dataset provided, will use random sampling for preferences")
        return

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.
        Either from dataset or from generated trajectories.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata
        """

        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_pref_sample_from_dataset()
            else:
                return self._create_pref_sample()

    def _create_pref_sample(self, chosen_traj: dict | None = None) -> PreferenceSample:
        """Create a preference prediction sample using various rejected trajectory generation strategies.

        Rewind Same Task
        - Creates a suboptimal trajectory by rewinding the chosen trajectory

        Suboptimal/Failure Same Task
        - Uses existing suboptimal/failure trajectories from the same task

        Different Task
        - Uses trajectories from completely different tasks

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata

        Raises:
            ValueError: If no chosen trajectories are available for preference generation
            RuntimeError: If all strategies fail and fallback rewind also fails
        """

        # Use provided chosen trajectory if given; otherwise sample one
        if chosen_traj is None:
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

        # Strategy selection with rebalancing on failure
        strategies = [
            (DataGenStrat.REWIND_SAME_TASK, self.preference_strategy_ratio[0]),
            (DataGenStrat.SUBOPTIMAL_SAME_TASK, self.preference_strategy_ratio[1]),
            (DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]),
            (
                DataGenStrat.VIDEO_BINNED,
                self.preference_strategy_ratio[3] if len(self.preference_strategy_ratio) > 3 else 0.0,
            ),
        ]

        # Remove strategies with zero probability
        strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

        max_attempts = 3  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while rejected_traj is None and attempt < max_attempts:
            attempt += 1

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                # All strategies have zero probability, fallback to rewind
                # Get success cutoff from pre-loaded map
                ds_key = chosen_traj["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
                
                rejected_traj = create_rewind_trajectory(
                    chosen_traj,
                    max_frames=self.config.max_frames,
                    use_embeddings=self.config.load_embeddings,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                strategy_used = DataGenStrat.REWIND_SAME_TASK
                break

            # Normalize probabilities
            normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

            # Select strategy based on rebalanced probabilities
            prob = random.random()
            cumulative_prob = 0.0
            selected_strategy = None

            for strat, normalized_prob in normalized_strategies:
                cumulative_prob += normalized_prob
                if prob <= cumulative_prob:
                    selected_strategy = strat
                    break

            # Execute selected strategy
            if selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                # Get success cutoff from pre-loaded map
                ds_key = chosen_traj["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
                
                rejected_traj = create_rewind_trajectory(
                    chosen_traj,
                    max_frames=self.config.max_frames,
                    use_embeddings=self.config.load_embeddings,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                strategy_used = DataGenStrat.REWIND_SAME_TASK

            elif selected_strategy == DataGenStrat.SUBOPTIMAL_SAME_TASK:
                rejected_traj = self._create_same_task_suboptimal_trajectory(chosen_traj)
                if rejected_traj is not None:
                    strategy_used = DataGenStrat.SUBOPTIMAL_SAME_TASK
                else:
                    # Strategy failed, remove it from future attempts
                    strategies = [
                        (strat, prob) for strat, prob in strategies if strat != DataGenStrat.SUBOPTIMAL_SAME_TASK
                    ]

            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                rejected_traj = self._create_different_task_trajectory(chosen_traj)
                if rejected_traj is not None:
                    strategy_used = DataGenStrat.DIFFERENT_TASK
                else:
                    # Strategy failed, remove it from future attempts
                    strategies = [(strat, prob) for strat, prob in strategies if strat != DataGenStrat.DIFFERENT_TASK]

            elif selected_strategy == DataGenStrat.VIDEO_BINNED:
                try:
                    chosen_traj, rejected_traj = self._create_video_binned_trajectory(
                        chosen_traj, num_bins=self.config.num_bins
                    )
                    strategy_used = DataGenStrat.VIDEO_BINNED
                except Exception as e:
                    rank_0_print(f"Video binning failed: {e}, removing from available strategies")
                    # Strategy failed, remove it from future attempts
                    strategies = [(strat, prob) for strat, prob in strategies if strat != DataGenStrat.VIDEO_BINNED]

        # Final fallback: If all strategies failed, use rewind
        if rejected_traj is None:
            # Get success cutoff from pre-loaded map
            ds_key = chosen_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
            
            rejected_traj = create_rewind_trajectory(
                chosen_traj,
                max_frames=self.config.max_frames,
                use_embeddings=self.config.load_embeddings,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            strategy_used = DataGenStrat.REWIND_SAME_TASK

        # ===============================================================
        # Subsample the chosen trajectory to max_frames
        # ===============================================================
        chosen_frames = None
        chosen_video_embeddings = None
        chosen_text_embedding = None

        rejected_frames = None
        rejected_video_embeddings = None
        rejected_text_embedding = None

        if self.config.load_embeddings and chosen_traj.get("embeddings_path"):
            chosen_video_embeddings = load_embeddings_from_path(chosen_traj["embeddings_path"], "video_embeddings")
            chosen_text_embedding = load_embeddings_from_path(chosen_traj["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = chosen_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                chosen_video_embeddings, self.config.max_frames
            )
            chosen_progress = compute_progress_from_segment(
                num_frames_total=len(chosen_video_embeddings),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            chosen_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            chosen_video_embeddings = subsampled
        else:
            if isinstance(chosen_traj["frames"], str):
                chosen_frames = load_frames_from_npz(chosen_traj["frames"])

            # Get success cutoff from pre-loaded map
            ds_key = chosen_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                chosen_frames, self.config.max_frames
            )
            chosen_progress = compute_progress_from_segment(
                num_frames_total=len(chosen_frames),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            chosen_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            chosen_frames = subsampled
        if "metadata" in chosen_traj:
            chosen_metadata.update(chosen_traj["metadata"])

        # ===============================================================
        # Subsample the rejected trajectory to max_frames
        # ===============================================================
        if self.config.load_embeddings and rejected_traj.get("embeddings_path"):
            rejected_video_embeddings = load_embeddings_from_path(rejected_traj["embeddings_path"], "video_embeddings")
            rejected_text_embedding = load_embeddings_from_path(rejected_traj["embeddings_path"], "text_embedding")

            if strategy_used == DataGenStrat.REWIND_SAME_TASK:
                rejected_video_embeddings = rejected_traj["frames"]
                rejected_progress = rejected_traj["target_progress"]
                rejected_metadata = rejected_traj["metadata"]
            else:
                # Get success cutoff from pre-loaded map
                ds_key = rejected_traj["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

                subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                    rejected_video_embeddings, self.config.max_frames
                )
                rejected_progress = compute_progress_from_segment(
                    num_frames_total=len(rejected_video_embeddings),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    frame_indices=indices,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                rejected_metadata = {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "subsampled_indices": indices,
                }
                rejected_video_embeddings = subsampled
        else:
            if isinstance(rejected_traj["frames"], str):
                rejected_frames = load_frames_from_npz(rejected_traj["frames"])
            else:
                rejected_frames = rejected_traj["frames"]

            if strategy_used == DataGenStrat.REWIND_SAME_TASK:
                rejected_frames = rejected_traj["frames"]
                rejected_progress = rejected_traj["target_progress"]
                rejected_metadata = rejected_traj["metadata"]
            else:
                # Get success cutoff from pre-loaded map
                ds_key = rejected_traj["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

                subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                    rejected_frames, self.config.max_frames
                )
                rejected_progress = compute_progress_from_segment(
                    num_frames_total=len(rejected_frames),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    frame_indices=indices,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                rejected_metadata = {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "subsampled_indices": indices,
                }
                rejected_frames = subsampled

        if "metadata" in rejected_traj:
            rejected_metadata.update(rejected_traj["metadata"])

        # Let's make sure to pad both trajectories to max_frames
        if self.config.load_embeddings:
            chosen_video_embeddings, chosen_progress = pad_trajectory_to_max_frames_torch(
                chosen_video_embeddings, chosen_progress, self.config.max_frames
            )
            rejected_video_embeddings, rejected_progress = pad_trajectory_to_max_frames_torch(
                rejected_video_embeddings, rejected_progress, self.config.max_frames
            )
        else:
            chosen_frames, chosen_progress = pad_trajectory_to_max_frames_np(
                chosen_frames, chosen_progress, self.config.max_frames
            )
            rejected_frames, rejected_progress = pad_trajectory_to_max_frames_np(
                rejected_frames, rejected_progress, self.config.max_frames
            )

        # If our strategy is different task, make sure the rejected trajectory has 0 progress
        if strategy_used == DataGenStrat.DIFFERENT_TASK:
            rejected_progress = [0.0] * len(rejected_progress)

        # Create preference sample structure
        sample = PreferenceSample(
            # Create Trajectory objects for chosen and rejected
            chosen_trajectory=Trajectory(
                frames=chosen_frames,
                frames_shape=chosen_frames.shape if chosen_frames is not None else None,
                video_embeddings=chosen_video_embeddings,
                text_embedding=chosen_text_embedding,
                id=chosen_traj["id"],
                task=chosen_traj["task"],
                lang_vector=chosen_traj["lang_vector"],
                data_source=chosen_traj["data_source"],
                quality_label=chosen_traj.get("quality_label"),
                is_robot=chosen_traj["is_robot"],
                target_progress=chosen_progress,
                data_gen_strategy=DataGenStrat.SUBSAMPLE_TASK.value,
                metadata=chosen_metadata,
            ),
            rejected_trajectory=Trajectory(
                frames=rejected_frames,
                frames_shape=rejected_frames.shape if rejected_frames is not None else None,
                video_embeddings=rejected_video_embeddings,
                text_embedding=rejected_text_embedding,
                id=rejected_traj["id"],
                task=rejected_traj["task"],
                lang_vector=rejected_traj["lang_vector"],
                data_source=rejected_traj["data_source"],
                quality_label=rejected_traj["quality_label"],
                is_robot=rejected_traj["is_robot"],
                target_progress=rejected_progress,
                data_gen_strategy=strategy_used.value,
                metadata=rejected_metadata,
            ),
        )
        return sample

    def _create_same_task_suboptimal_trajectory(self, chosen_traj: dict) -> dict | None:
        """Create a suboptimal trajectory from the same task as the chosen trajectory.

        This function tries to find an existing suboptimal/failure trajectory from the same task.
        Returns None if no suboptimal trajectories are available.

        Args:
            chosen_traj: The chosen (optimal) trajectory dictionary

        Returns:
            Optional[Dict]: The rejected trajectory, or None if none available
        """
        task_name = chosen_traj["task"]

        # Try to find suboptimal trajectories from the same task
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        same_task_suboptimal = [
            self.dataset[idx] for idx in same_task_suboptimal_indices if self.dataset[idx]["id"] != chosen_traj["id"]
        ]

        if same_task_suboptimal:
            return random.choice(same_task_suboptimal)
        else:
            return None

    def _create_different_task_trajectory(self, chosen_traj: dict) -> dict | None:
        """Create a trajectory from a different task than the chosen trajectory.

        This function tries to find trajectories from different tasks.
        Returns None if no other tasks are available.

        Args:
            chosen_traj: The chosen trajectory dictionary

        Returns:
            Optional[Dict]: The rejected trajectory, or None if none available
        """
        # Find other tasks
        other_tasks = [task for task in self.optimal_by_task.keys() if task != chosen_traj["task"]]
        if other_tasks:
            other_task = random.choice(other_tasks)
            other_task_indices = self.optimal_by_task[other_task]

            if other_task_indices:
                other_idx = random.choice(other_task_indices)
                other_traj = self.dataset[other_idx]

                # Check if it's not the same trajectory
                if other_traj["id"] != chosen_traj["id"]:
                    return other_traj
        else:
            # Only one task available
            return None

        return None
