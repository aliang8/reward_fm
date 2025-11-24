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
        rank_0_print(
            f"[PREF SAMPLER] No suboptimal/failure data available; skipping suboptimal strategy for preferences. Has suboptimal: {self._has_suboptimal}"
        )

        # Initialize preference dataset
        self._load_preference_dataset()

    def _generate_sample(self, item: dict):
        return self._create_pref_sample(item)

    def _create_pref_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            return None

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        return None

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_print("[PREF SAMPLER] No preference dataset provided, will use random sampling for preferences")
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
                return None

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
            strategies.append((DataGenStrat.REWOUND, self.preference_strategy_ratio[0]))
        if self._has_suboptimal and self.preference_strategy_ratio[1] > 0:
            strategies.append((DataGenStrat.SUBOPTIMAL, self.preference_strategy_ratio[1]))
        if self.preference_strategy_ratio[2] > 0:
            strategies.append((DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]))

        max_attempts = 10  # Limit retry attempts to prevent infinite loops
        max_strategy_attempts = 3  # Maximum attempts per strategy before removing it
        attempt = 0

        # Track attempts per strategy
        strategy_attempt_counts = {strat: 0 for strat, _ in strategies}

        while rejected_traj is None and attempt < max_attempts:
            attempt += 1

            # Check if we have any strategies left
            if not strategies:
                return None

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                return None

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

            if selected_strategy == DataGenStrat.REWOUND:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_rewound_traj(chosen_traj)
                    if rejected_traj is not None:
                        break
            elif selected_strategy == DataGenStrat.SUBOPTIMAL:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_same_task_suboptimal(chosen_traj)
                    if rejected_traj is not None:
                        break
            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                rejected_traj = None
                for _ in range(max_retries):
                    rejected_traj = self._get_different_video_traj(chosen_traj)
                    if rejected_traj is not None:
                        break
            else:
                return None

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

        # If we still don't have a sample after all attempts, return None
        if rejected_traj is None:
            rank_0_print(f"[PREF SAMPLER] Failed to generate preference sample after {max_attempts} attempts - all strategies exhausted")
            return None

        chosen_trajectory = self._get_traj_from_data(chosen_traj)
        rejected_trajectory = self._get_traj_from_data(rejected_traj)

        # If our strategy is different task, make sure the rejected trajectory has 0 progress
        if strategy_used in [
            DataGenStrat.DIFFERENT_TASK,
            DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
            DataGenStrat.SUBOPTIMAL,
        ]:
            rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)

        # Create preference sample structure
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=strategy_used.value,
        )
        sample.resample_attempts = attempt
        return sample
