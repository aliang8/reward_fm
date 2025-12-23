from typing import Dict, Any

import random
import torch

from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    DataGenStrat,
    load_embeddings_from_path,
)
from rfm.utils.distributed import rank_0_print
from rfm.utils.logger import get_logger

logger = get_logger()


class ProgressSampler(RFMBaseSampler):
    """Data generator for progress samples."""

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

    def _generate_sample(self, item: Dict[str, Any]):
        return self._create_progress_sample(item)

    def _create_progress_sample(self, traj: Dict[str, Any]):
        """Create a progress sample using normalized and rebalanced strategy selection.

        Implements six strategies:
        1. Successful: Linspace subsample with end_idx between cutoff and total
        2. Rewind: Create rewound trajectory from same task
        3. Different Task: Use trajectory from different task (progress set to 0.0)
        4. Subsequence: Segment subsampling (same as previous default)
        5. Reverse Progress: Same as subsequence but reverses frames and progress targets
        6. Uniform Sample: Sample two random frames, use them as segment bounds, then linspace subsample
        """
        # Initialize variables for strategy selection
        processed_traj = None
        strategy_used = None
        subsample_strategy = None

        # Strategy setup with rebalancing on failure
        # [successful, rewind, different_task_instruction, subsequence, reverse_progress, uniform_sample]
        strategies = [
            (DataGenStrat.SUCCESSFUL, self.config.progress_strategy_ratio[0]),
            (DataGenStrat.REWOUND, self.config.progress_strategy_ratio[1]),
            (
                DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
                self.config.progress_strategy_ratio[2] if len(self.config.progress_strategy_ratio) > 2 else 0.0,
            ),
            (
                DataGenStrat.SUBSEQUENCE,
                self.config.progress_strategy_ratio[3] if len(self.config.progress_strategy_ratio) > 3 else 0.0,
            ),
            (
                DataGenStrat.REVERSE_PROGRESS,
                self.config.progress_strategy_ratio[4] if len(self.config.progress_strategy_ratio) > 4 else 0.0,
            ),
            (
                DataGenStrat.UNIFORM_SAMPLE,
                self.config.progress_strategy_ratio[5] if len(self.config.progress_strategy_ratio) > 5 else 0.0,
            ),
        ]

        if self.config.pairwise_progress:
            # remove rewind same task strategy for pairwise progress
            strategies[1] = (
                DataGenStrat.REWOUND,
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

            # Execute selected strategy
            if selected_strategy == DataGenStrat.SUCCESSFUL:
                processed_traj = traj
                subsample_strategy = "successful"
            elif selected_strategy == DataGenStrat.SUBSEQUENCE:
                processed_traj = traj
                subsample_strategy = "subsequence"
            elif selected_strategy == DataGenStrat.REVERSE_PROGRESS:
                processed_traj = traj
                subsample_strategy = "reverse_progress"
            elif selected_strategy == DataGenStrat.UNIFORM_SAMPLE:
                processed_traj = traj
                subsample_strategy = "uniform_sample"
            elif selected_strategy == DataGenStrat.REWOUND:
                processed_traj = self._get_rewound_traj(traj)
                subsample_strategy = None  # Rewound trajectories are already processed
            elif selected_strategy == DataGenStrat.DIFFERENT_TASK_INSTRUCTION:
                processed_traj = self._get_different_task_instruction(traj)
                subsample_strategy = None  # Different task instruction uses same trajectory with different task
            else:
                return None

            # Check if strategy succeeded
            if processed_traj is not None:
                strategy_used = selected_strategy
            else:
                # Remove failed strategy and try again
                strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                continue

        # If we still don't have a sample after all attempts, return None
        if processed_traj is None:
            logger.trace(
                f"[PROGRESS SAMPLER] Failed to generate progress sample after {max_attempts} attempts - all strategies exhausted"
            )
            return None

        progress_traj = self._get_traj_from_data(processed_traj, subsample_strategy=subsample_strategy)

        # Handle special cases
        if strategy_used in [DataGenStrat.DIFFERENT_TASK, DataGenStrat.DIFFERENT_TASK_INSTRUCTION]:
            # We need to use the original task embeddings instead of the different task embeddings
            if self.config.load_embeddings and traj.get("embeddings_path"):
                progress_traj.text_embedding = load_embeddings_from_path(traj["embeddings_path"])["text_embedding"]
            progress_traj.lang_vector = traj["lang_vector"]
            progress_traj.task = traj["task"]
            progress_traj.target_progress = [0.0] * len(progress_traj.target_progress)

        strategy_value = strategy_used.value if isinstance(strategy_used, DataGenStrat) else strategy_used
        sample = ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
            data_gen_strategy=strategy_value,
        )
        sample.resample_attempts = attempt
        return sample
