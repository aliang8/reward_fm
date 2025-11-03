import random
import torch

from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    DataGenStrat,
    load_embeddings_from_path,
)


class ProgressSampler(RFMBaseSampler):
    """Data generator for progress samples."""

    def __init__(self, config, dataset, combined_indices, dataset_success_cutoff_map=None, is_evaluation=False, verbose=True, **kwargs):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

    def _generate_sample(self, item: dict):
        return self._create_progress_sample(item)

    def _create_progress_sample(self, traj: dict):
        """Create a progress sample using normalized and rebalanced strategy selection.

        Implements three strategies:
        1. Successful: Use original trajectory as-is
        2. Rewind Same Task: Create rewound trajectory from same task
        3. Different Task: Use trajectory from different task (progress set to 0.0)
        """

        # Initialize variables for strategy selection
        processed_traj = None
        strategy_used = None

        # Strategy setup with rebalancing on failure
        strategies = [
            ("successful", self.config.progress_strategy_ratio[0]),
            (DataGenStrat.REWIND_SAME_TASK, self.config.progress_strategy_ratio[1]),
            (DataGenStrat.DIFFERENT_TASK, self.config.progress_strategy_ratio[2]),
        ]

        if self.config.pairwise_progress:
            # remove rewind same task strategy for pairwise progress
            strategies[1] = (
                DataGenStrat.REWIND_SAME_TASK,
                0.0,
            )

        # Remove strategies with zero probability
        strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

        max_attempts = 10  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while processed_traj is None and attempt < max_attempts:
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

            # Execute selected strategy
            if selected_strategy == "successful":
                processed_traj = traj.copy()
            elif selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                processed_traj = self._get_rewound_traj(traj)
            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                processed_traj = self._get_different_task(traj)
            else:
                raise ValueError(f"Invalid strategy selected: {selected_strategy}")

            # Check if strategy succeeded
            if processed_traj is not None:
                strategy_used = selected_strategy
            else:
                # Remove failed strategy and try again
                strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                continue

        # If we still don't have a sample after all attempts, raise an error
        if processed_traj is None:
            raise ValueError(
                f"Failed to generate progress sample after {max_attempts} attempts - all strategies exhausted"
            )

        progress_traj = self._get_traj_from_data(processed_traj)

        # Handle special cases
        if strategy_used == DataGenStrat.DIFFERENT_TASK:
            # We need to use the original task embeddings instead of the different task embeddings
            if self.config.load_embeddings and traj.get("embeddings_path"):
                progress_traj.text_embedding = load_embeddings_from_path(traj["embeddings_path"], "text_embedding")
            progress_traj.lang_vector = traj["lang_vector"]
            progress_traj.task = traj["task"]
            progress_traj.target_progress = [0.0] * len(progress_traj.target_progress)

        strategy_value = strategy_used.value if isinstance(strategy_used, DataGenStrat) else strategy_used
        return ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
            data_gen_strategy=strategy_value,
        )
