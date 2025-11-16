#!/usr/bin/env python3
"""
PrefSampler class for producing batches of preference data.
"""

import random
import torch

from rfm.data.dataset_types import PreferenceSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    DataGenStrat,
    load_frames_from_npz,
    linspace_subsample_frames,
)
from rfm.utils.distributed import rank_0_print
from rfm.utils.timer import timer


class PrefSampler(RFMBaseSampler):
    """Data generator for producing batches of preference prediction data."""

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        self.dataset_preference_ratio = config.dataset_preference_ratio
        self.preference_strategy_ratio: list[float] = config.preference_strategy_ratio
        self._has_suboptimal = any(indices for indices in self.suboptimal_by_task.values())
        if verbose and self.preference_strategy_ratio[1] > 0 and not self._has_suboptimal:
            rank_0_print("No suboptimal/failure data available; skipping suboptimal strategy for preferences.")

        # Initialize preference dataset
        self._load_preference_dataset()

    def _generate_sample(self, item: dict):
        return self._create_pref_sample(item)

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
            chosen_progress.append(i / (len(frames_data) - chosen_start - 1))

        # Extract frames from the rejected bin (this will be the "rejected" trajectory)
        rejected_start, rejected_end = bin_boundaries[rejected_bin_idx]
        rejected_frames = frames_data[rejected_start:rejected_end]

        rejected_progress = []
        for i in range(len(rejected_frames)):
            rejected_progress.append(i / (len(frames_data) - rejected_start - 1))

        # Apply uniform subsampling to both bins to ensure consistent frame counts
        # Use uniform subsampling for real trajectories (not rewound)
        num_frames_to_sample = self.config.max_frames
        chosen_frames, chosen_indices = linspace_subsample_frames(chosen_frames, num_frames_to_sample)
        rejected_frames, rejected_indices = linspace_subsample_frames(rejected_frames, num_frames_to_sample)

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
        strategies = []
        if self.preference_strategy_ratio[0] > 0:
            strategies.append((DataGenStrat.REWIND_SAME_TASK, self.preference_strategy_ratio[0]))
        if self._has_suboptimal and self.preference_strategy_ratio[1] > 0:
            strategies.append((DataGenStrat.SUBOPTIMAL_SAME_TASK, self.preference_strategy_ratio[1]))
        if self.preference_strategy_ratio[2] > 0:
            strategies.append((DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]))
        if len(self.preference_strategy_ratio) > 3 and self.preference_strategy_ratio[3] > 0:
            strategies.append((DataGenStrat.VIDEO_BINNED, self.preference_strategy_ratio[3]))

        max_attempts = 10  # Limit retry attempts to prevent infinite loops
        max_strategy_attempts = 3  # Maximum attempts per strategy before removing it
        attempt = 0

        # Track attempts per strategy
        strategy_attempt_counts = {strat: 0 for strat, _ in strategies}

        while rejected_traj is None and attempt < max_attempts:
            attempt += 1

            # Check if we have any strategies left
            if not strategies:
                raise ValueError("No strategies available - all strategies failed to generate samples")

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                raise ValueError("No strategies with positive probability available")

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

            # Execute selected strategy with retry logic
            max_retries = 3  # Number of retry attempts for sampling

            if selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_rewound_traj(chosen_traj)
                    if rejected_traj is not None:
                        break
            elif selected_strategy == DataGenStrat.SUBOPTIMAL_SAME_TASK:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_same_task_suboptimal(chosen_traj)
                    if rejected_traj is not None:
                        break
            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_different_task(chosen_traj)
                    if rejected_traj is not None:
                        break
            elif selected_strategy == DataGenStrat.VIDEO_BINNED:
                rejected_traj = None
                for _ in range(max_retries):
                    try:
                        chosen_traj, rejected_traj = self._create_video_binned_trajectory(
                            chosen_traj, num_bins=self.config.num_bins
                        )
                        if rejected_traj is not None:
                            break
                    except Exception as e:
                        rank_0_print(f"Video binning failed: {e}")
                        rejected_traj = None
            else:
                raise ValueError(f"Invalid strategy selected: {selected_strategy}")

            # Check if strategy succeeded
            if rejected_traj is not None:
                strategy_used = selected_strategy
            else:
                # Strategy failed - increment attempt count
                strategy_attempt_counts[selected_strategy] = strategy_attempt_counts.get(selected_strategy, 0) + 1

                # Only remove strategy if it has failed max_strategy_attempts times
                if strategy_attempt_counts[selected_strategy] >= max_strategy_attempts:
                    strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                    continue

        # If we still don't have a sample after all attempts, raise an error
        if rejected_traj is None:
            raise ValueError(
                f"Failed to generate preference sample after {max_attempts} attempts - all strategies exhausted"
            )

        chosen_trajectory = self._get_traj_from_data(chosen_traj)
        rejected_trajectory = self._get_traj_from_data(rejected_traj)

        # If our strategy is different task, make sure the rejected trajectory has 0 progress
        if strategy_used == DataGenStrat.DIFFERENT_TASK:
            rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)

        # Create preference sample structure
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=strategy_used.value,
        )
        sample.resample_attempts = attempt
        return sample
