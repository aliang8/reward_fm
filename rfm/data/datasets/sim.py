#!/usr/bin/env python3


import random

from rfm.data.dataset_types import SimilaritySample, Trajectory
from rfm.data.datasets.base import RFMBaseDataset
from rfm.data.datasets.helpers import DataGenStrat
from rfm.utils.distributed import rank_0_print


class SimilarityDataset(RFMBaseDataset):
    """Data generator for producing batches for similarity scoring."""

    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        super().__init__(config, is_evaluation, verbose=verbose, **kwargs)
        self.similarity_strategy_ratio: list[float] = config.similarity_strategy_ratio

        if self.verbose:
            rank_0_print(f"SimilarityDataset initialized with {len(self.dataset)} total trajectories")

    def __getitem__(self, idx):
        return self._create_similarity_sample()

    def _create_similarity_sample(self, ref_traj: dict | None = None) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.

        Two modes:
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
            - here o^1 is preferred and should be ranked higher than o^2
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
            - here o^1 is preferred and should be ranked higher than o^2

        Args:
            ref_traj: Optional reference trajectory. If None, samples from optimal trajectories.
        """

        # Use provided reference trajectory if given; otherwise sample one
        if ref_traj is None:
            # Use preprocessed optimal trajectories from index maps
            if not self.optimal_by_task:
                raise ValueError("No optimal trajectories found for similarity sample generation")

            # Get a random task and optimal trajectory from it
            task_name = random.choice(list(self.optimal_by_task.keys()))
            optimal_indices = self.optimal_by_task[task_name]
            while not optimal_indices:
                task_name = random.choice(list(self.optimal_by_task.keys()))
                optimal_indices = self.optimal_by_task[task_name]

            optimal_idx = random.choice(optimal_indices)
            ref_traj = self.dataset[optimal_idx]

        traj_sim, traj_diff = None, None
        strategy_used = None

        # Strategy selection with rebalancing on failure
        strategies = [
            (DataGenStrat.REWIND_SAME_TASK, self.similarity_strategy_ratio[0]),
            (DataGenStrat.SUBOPTIMAL_SAME_TASK, self.similarity_strategy_ratio[1]),
        ]

        # Remove strategies with zero probability
        strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

        max_attempts = 3  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while traj_sim is None and attempt < max_attempts:
            attempt += 1

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                # All strategies have zero probability, fallback to rewind
                traj_sim, traj_diff = self._get_traj_dicts_for_rewind(ref_traj)
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
                traj_sim, traj_diff = self._get_traj_dicts_for_rewind(ref_traj)
                strategy_used = DataGenStrat.REWIND_SAME_TASK

            elif selected_strategy == DataGenStrat.SUBOPTIMAL_SAME_TASK:
                result = self._get_traj_dicts_for_suboptimal(ref_traj)
                if result is not None:
                    traj_sim, traj_diff = result
                    strategy_used = DataGenStrat.SUBOPTIMAL_SAME_TASK
            else:
                raise ValueError(f"Invalid strategy selected: {selected_strategy}")

        # Final fallback: If all strategies failed, use rewind
        if traj_sim is None:
            traj_sim, traj_diff = self._get_traj_dicts_for_rewind(ref_traj)
            strategy_used = DataGenStrat.REWIND_SAME_TASK

        return SimilaritySample(
            ref_trajectory=self._get_traj_from_data(ref_traj),
            sim_trajectory=self._get_traj_from_data(traj_sim),
            diff_trajectory=self._get_traj_from_data(traj_diff),
            data_gen_strategy=strategy_used.value,
        )

    def _get_traj_dicts_for_rewind(self, ref_traj: dict) -> tuple[dict | Trajectory, dict]:
        """Get traj_sim and traj_diff for rewind strategy.

        Two cases:
        1) sim = rewound, diff = different task
        2) sim = same task optimal, diff = rewound

        Args:
            ref_traj: Reference trajectory

        Returns:
            Tuple of (traj_sim, traj_diff) where both can be dict or Trajectory objects
        """
        # Try case 1: sim = rewound, diff = different task
        traj_sim = self._get_rewound_traj(ref_traj)
        traj_diff = self._get_different_task(ref_traj)

        if traj_diff is not None:
            return traj_sim, traj_diff

        # Case 1 failed, try case 2: sim = same task optimal, diff = rewound
        traj_sim = self._get_same_task_optimal(ref_traj)
        if traj_sim is None:
            # No same task optimal available, fallback: both rewound
            traj_sim = self._get_rewound_traj(ref_traj)
            traj_diff = self._get_rewound_traj(ref_traj)
            return traj_sim, traj_diff

        traj_diff = self._get_rewound_traj(ref_traj)
        return traj_sim, traj_diff

    def _get_traj_dicts_for_suboptimal(self, ref_traj: dict) -> tuple[dict, dict | Trajectory] | None:
        """Get traj_sim and traj_diff for suboptimal strategy.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Tuple of (traj_sim, traj_diff) or None if not available. Both can be dict or Trajectory objects.
        """
        traj_sim = self._get_same_task_optimal(ref_traj)
        if traj_sim is None:
            return None

        traj_diff = self._get_same_task_suboptimal(ref_traj)
        if traj_diff is None:
            # Fallback: use rewound for diff
            traj_diff = self._get_rewound_traj(ref_traj)
            return traj_sim, traj_diff

        return traj_sim, traj_diff
