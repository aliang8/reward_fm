import random

from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.datasets.base import RFMBaseDataset
from rfm.data.datasets.helpers import (
    DataGenStrat,
    load_embeddings_from_path,
)


class ProgressDataset(RFMBaseDataset):
    """Data generator for progress samples."""

    def __getitem__(self, idx):
        """Iterate over one sample per trajectory in the dataset."""
        dataset_len = len(self.dataset)
        traj = self.dataset[idx % dataset_len]
        sample = self._create_progress_sample(traj)
        return sample

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

        max_attempts = 3  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while processed_traj is None and attempt < max_attempts:
            attempt += 1

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                # All strategies have zero probability, fallback to successful
                processed_traj = traj.copy()
                strategy_used = "successful"
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
            if selected_strategy == "successful":
                processed_traj = traj.copy()
                strategy_used = "successful"

            elif selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                processed_traj = self._get_rewound_traj(traj)
                strategy_used = DataGenStrat.REWIND_SAME_TASK

            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                other_traj = self._get_different_task(traj)
                if other_traj is not None:
                    processed_traj = other_traj
                    strategy_used = DataGenStrat.DIFFERENT_TASK
                else:
                    # Strategy failed, remove it from future attempts
                    strategies = [(strat, prob) for strat, prob in strategies if strat != DataGenStrat.DIFFERENT_TASK]

        # Final fallback: If all strategies failed, use successful
        if processed_traj is None:
            processed_traj = traj.copy()
            strategy_used = "successful"

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
