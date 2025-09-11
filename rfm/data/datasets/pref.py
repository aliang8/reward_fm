#!/usr/bin/env python3
"""
PrefDataset class for producing batches of preference prediction data.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from rfm.data.dataset_types import PreferenceSample, ProgressSample, Trajectory
from rfm.data.datasets.base import RFMBaseDataset
from rfm.utils.logging import rank_0_print, timer
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    randomly_subsample_frames,
    pad_trajectory_to_max_frames,
    subsample_frames_and_progress,
    create_rewind_trajectory,
    load_frames_from_npz,
    DataGenStrat,
)


class PrefDataset(RFMBaseDataset):
    """Data generator for producing batches of preference prediction data."""

    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        """Initialize PrefDataset with configuration."""
        self.dataset_preference_ratio = config.dataset_preference_ratio
        self.preference_strategy_ratio: List[float] = config.preference_strategy_ratio

        super().__init__(config, is_evaluation, verbose=verbose, **kwargs)

        # Initialize preference dataset
        self._load_preference_dataset()

        rank_0_print(f"PrefDataset initialized with {len(self.dataset)} total trajectories")

        # We only want to iterate over non-failure and suboptimal trajectories
        self.filtered_ds = self.dataset.filter(lambda x: x["quality_label"] not in ["failure", "suboptimal"])

    def __len__(self):
        return len(self.filtered_ds)

    def __getitem__(self, idx):
        """Iterate over one sample per trajectory in the dataset."""
        dataset_len = len(self.filtered_ds)
        chosen_traj = self.filtered_ds[idx % dataset_len]
        sample = self._create_preference_sample_with_strategies(chosen_traj)
        return sample

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

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.

        This method can create preference samples from two sources:

        **Dataset Source:**
        - Uses pre-existing preference data from the loaded preference dataset
        - Good for learning from curated, high-quality preference examples
        - Controlled by config.dataset_preference_ratio

        **Data Augmentation Strategies:**
        When not using dataset preferences, delegates to _create_preference_sample_with_strategies()
        which implements various strategies for generating rejected trajectories.

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

    def _create_preference_sample_with_strategies(self, chosen_traj: Optional[Dict] = None) -> PreferenceSample:
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

        prob = random.random()
        if prob < self.preference_strategy_ratio[0]:
            # Strategy 1: Use rewind-generated suboptimal trajectory from same task
            rejected_traj = create_rewind_trajectory(chosen_traj, max_frames=self.config.max_frames)
            strategy_used = DataGenStrat.REWIND_SAME_TASK

        elif prob < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1]:
            # Strategy 2: Use random suboptimal trajectory from same task
            rejected_traj = self._create_same_task_suboptimal_trajectory(chosen_traj)
            if rejected_traj is not None:
                strategy_used = DataGenStrat.SUBOPTIMAL_SAME_TASK

        elif (
            prob
            < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1] + self.preference_strategy_ratio[2]
        ):
            # Strategy 3: Use trajectory from different task (can be chosen or suboptimal)
            rejected_traj = self._create_different_task_trajectory(chosen_traj)
            if rejected_traj is not None:
                strategy_used = DataGenStrat.DIFFERENT_TASK

        else:
            # Strategy 4: Create preference sample from different bins of the same video
            try:
                chosen_traj, rejected_traj = self._create_video_binned_trajectory(
                    chosen_traj, num_bins=self.config.num_bins
                )
                strategy_used = DataGenStrat.VIDEO_BINNED
            except Exception as e:
                rank_0_print(f"Video binning failed: {e}, will fall back to rewind")

        # Fallback: If any strategy failed to produce a rejected trajectory, use rewind
        if rejected_traj is None:
            rejected_traj = create_rewind_trajectory(chosen_traj, max_frames=self.config.max_frames)
            strategy_used = DataGenStrat.REWIND_SAME_TASK

        # ===============================================================
        # Subsample the chosen trajectory to max_frames
        # ===============================================================
        if isinstance(chosen_traj["frames"], str):
            chosen_traj["frames"] = load_frames_from_npz(chosen_traj["frames"])

        chosen_frames, chosen_progress, chosen_metadata = subsample_frames_and_progress(
            chosen_traj["frames"], self.config.max_frames
        )
        if "metadata" in chosen_traj:
            chosen_metadata.update(chosen_traj["metadata"])

        # ===============================================================
        # Subsample the rejected trajectory to max_frames
        # ===============================================================

        if isinstance(rejected_traj["frames"], str):
            rejected_traj["frames"] = load_frames_from_npz(rejected_traj["frames"])

        if strategy_used != DataGenStrat.REWIND_SAME_TASK:
            # try subsampling the rejected trajectory
            rejected_frames, rejected_progress, rejected_metadata = subsample_frames_and_progress(
                rejected_traj["frames"], self.config.max_frames
            )
            if "metadata" in rejected_traj:
                rejected_metadata.update(rejected_traj["metadata"])

        else:
            rejected_frames = rejected_traj["frames"]
            rejected_progress = rejected_traj["target_progress"]
            rejected_metadata = rejected_traj["metadata"]

        # Let's make sure to pad both trajectories to max_frames
        chosen_frames, chosen_progress = pad_trajectory_to_max_frames(
            chosen_frames, chosen_progress, self.config.max_frames
        )
        rejected_frames, rejected_progress = pad_trajectory_to_max_frames(
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
                frames_shape=chosen_frames.shape,
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
                frames_shape=rejected_frames.shape,
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

    def _create_same_task_suboptimal_trajectory(self, chosen_traj: Dict) -> Optional[Dict]:
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

    def _create_different_task_trajectory(self, chosen_traj: Dict) -> Optional[Dict]:
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
