#!/usr/bin/env python3
from typing import Dict, List, Tuple, Optional, Union, Any

import torch

from rfm.data.dataset_types import SimilaritySample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import DataGenStrat
from rfm.data.dataset_category import is_failure_ds, is_paired_ds
from rfm.utils.logger import get_logger, rank_0_info

logger = get_logger()


class SimSampler(RFMBaseSampler):
    """Data generator for producing batches for similarity scoring."""

    def __init__(
        self,
        is_evaluation=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_strategy_ratio: List[float] = self.config.similarity_strategy_ratio
        self._has_paired_human_robot = (
            any(
                len(entry.get("robot", [])) > 0 and len(entry.get("human", [])) > 0
                for entry in self.paired_human_robot_by_task.values()
            )
            if self.paired_human_robot_by_task
            else False
        )
        self._has_suboptimal = (
            any(len(indices) > 0 for indices in self.suboptimal_by_task.values()) if self.suboptimal_by_task else False
        )
        rank_0_info(
            f"[SIM SAMPLER] Has paired human/robot: {self._has_paired_human_robot}, Has suboptimal: {self._has_suboptimal}"
        )

    def _generate_sample(self, item: dict, preferred_strategy: Optional[DataGenStrat] = None):
        return self._create_similarity_sample(ref_traj=item, preferred_strategy=preferred_strategy)

    def _execute_strategy(
        self, strategy: DataGenStrat, ref_traj: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
        """Execute a strategy to get trajectory pairs.
        
        Args:
            strategy: The strategy to execute
            ref_traj: The reference trajectory
            
        Returns:
            Tuple of (traj_sim, traj_diff) or None if failed
        """
        if strategy == DataGenStrat.REWIND:
            return self._get_traj_dicts_for_rewind(ref_traj)
        elif strategy == DataGenStrat.SUBOPTIMAL:
            return self._get_traj_dicts_for_suboptimal(ref_traj)
        elif strategy == DataGenStrat.PAIRED_HUMAN_ROBOT:
            return self._get_traj_dicts_for_paired_human_robot(ref_traj)
        else:
            return None

    def _create_similarity_sample(
        self, ref_traj: Optional[Dict[str, Any]] = None, preferred_strategy: Optional[DataGenStrat] = None
    ) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.

        Two modes:
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
            - here o^1 is preferred and should be ranked higher than o^2
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
            - here o^1 is preferred and should be ranked higher than o^2

        Args:
            ref_traj: Optional reference trajectory. If None, samples from optimal trajectories.
        """
        # Log when similarity sampler is called
        traj_id = ref_traj.get("id", "unknown") if ref_traj is not None else "sampling_new"
        logger.trace(f"[SIM SAMPLER] Creating similarity sample for trajectory ID: {traj_id}")

        # Use provided reference trajectory if given; otherwise sample one
        if ref_traj is None:
            # Use preprocessed optimal trajectories from index maps
            if not self.optimal_by_task:
                return None

            # Filter out tasks with empty optimal_indices to avoid infinite loop
            valid_tasks = {
                task: indices
                for task, indices in self.optimal_by_task.items()
                if indices  # Only include tasks with non-empty indices
            }

            if not valid_tasks:
                # No valid tasks with optimal trajectories available
                return None

            # Get a random task and optimal trajectory from it
            task_name = self._local_random.choice(list(valid_tasks.keys()))
            optimal_indices = valid_tasks[task_name]

            # Double-check that we have valid indices (should always be true now)
            if not optimal_indices:
                return None

            optimal_idx = self._local_random.choice(optimal_indices)
            ref_traj = self.dataset[optimal_idx]

        # Check if ref_traj is successful - if not, return None to try a different trajectory
        quality_label = ref_traj.get("quality_label")
        partial_success = ref_traj.get("partial_success")
        data_source = ref_traj.get("data_source", "")
        is_roboarena = partial_success is not None and data_source and "roboarena" in str(data_source).lower()

        if is_roboarena:
            # For RoboArena, require partial_success to exist
            if partial_success is None:
                logger.trace(
                    f"[SIM SAMPLER] Ref trajectory {ref_traj.get('id', 'unknown')} missing partial_success, skipping"
                )
                return None
        else:
            # For non-RoboArena, require quality_label to be "successful"
            if quality_label != "successful":
                logger.trace(
                    f"[SIM SAMPLER] Ref trajectory {ref_traj.get('id', 'unknown')} is not successful (quality_label: {quality_label}), skipping"
                )
                return None

        traj_sim, traj_diff = None, None
        strategy_used = None
        is_failure_source = is_failure_ds(data_source) if data_source else False
        is_paired_source = is_paired_ds(data_source) if data_source else False

        # Strategy selection: use preferred_strategy if provided, otherwise select based on ratios
        if preferred_strategy is not None:
            # Use the preferred strategy directly
            logger.trace(
                f"[SIM SAMPLER] Using preferred strategy: {preferred_strategy.value}"
            )
            result = self._execute_strategy(preferred_strategy, ref_traj)
            if result is None:
                logger.trace(
                    f"[SIM SAMPLER] Preferred strategy {preferred_strategy.value} failed, returning None"
                )
                return None
            traj_sim, traj_diff = result
            strategy_used = preferred_strategy
        else:
            # Strategy selection with data_source-based filtering and boosting
            strategies = []

            # Always include REWIND if ratio > 0
            if self.similarity_strategy_ratio[0] > 0:
                strategies.append((DataGenStrat.REWIND, self.similarity_strategy_ratio[0]))

            # SUBOPTIMAL: include if data_source is in failure category
            if len(self.similarity_strategy_ratio) > 1 and self.similarity_strategy_ratio[1] > 0 and is_failure_source:
                # Boost probability by 2x if data_source is in failure category
                boosted_prob = self.similarity_strategy_ratio[1] * 2.0
                strategies.append((DataGenStrat.SUBOPTIMAL, boosted_prob))

            # PAIRED_HUMAN_ROBOT: only include if data_source is in paired category
            if (
                self._has_paired_human_robot
                and len(self.similarity_strategy_ratio) > 2
                and self.similarity_strategy_ratio[2] > 0
                and is_paired_source
            ):
                # Boost probability by 2x if data_source is in paired category
                boosted_prob = self.similarity_strategy_ratio[2] * 2.0
                strategies.append((DataGenStrat.PAIRED_HUMAN_ROBOT, boosted_prob))

            # Remove strategies with zero probability
            strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

            max_attempts = 10  # Limit retry attempts to prevent infinite loops
            max_strategy_attempts = 4  # Maximum attempts per strategy before removing it
            attempt = 0

            strategies_tried = []
            # Track attempts per strategy
            strategy_attempt_counts = {strat: 0 for strat, _ in strategies}

            while traj_sim is None and attempt < max_attempts:
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
                prob = self._local_random.random()
                cumulative_prob = 0.0
                selected_strategy = None

                for strat, normalized_prob in normalized_strategies:
                    cumulative_prob += normalized_prob
                    if prob <= cumulative_prob:
                        selected_strategy = strat
                        strategies_tried.append(selected_strategy)
                        break

                # Log strategy attempt
                logger.trace(
                    f"[SIM SAMPLER] Attempt {attempt}/{max_attempts}: Trying strategy {selected_strategy.value if selected_strategy else 'None'}"
                )

                # Execute selected strategy
                result = self._execute_strategy(selected_strategy, ref_traj)
                if result is not None:
                    traj_sim, traj_diff = result
                    strategy_used = selected_strategy
                    logger.trace(f"[SIM SAMPLER] Strategy {selected_strategy.value} succeeded on attempt {attempt}")
                else:
                    # Strategy failed - increment attempt count
                    strategy_attempt_counts[selected_strategy] = strategy_attempt_counts.get(selected_strategy, 0) + 1
                    failed_count = strategy_attempt_counts[selected_strategy]

                    logger.trace(
                        f"[SIM SAMPLER] Strategy {selected_strategy.value} failed (failure count: {failed_count}/{max_strategy_attempts})"
                    )

                    # Only remove strategy if it has failed max_strategy_attempts times
                    if strategy_attempt_counts[selected_strategy] >= max_strategy_attempts:
                        logger.trace(
                            f"[SIM SAMPLER] Removing strategy {selected_strategy.value} after {max_strategy_attempts} consecutive failures"
                        )
                        strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                        continue

            # If we still don't have a sample after all attempts, return None
            if traj_sim is None or traj_diff is None:
                logger.trace(
                    f"[SIM SAMPLER] Failed to generate similarity sample after {max_attempts} attempts - all strategies exhausted"
                )
                return None

        sample = SimilaritySample(
            ref_trajectory=self._get_traj_from_data(ref_traj),
            sim_trajectory=self._get_traj_from_data(traj_sim),
            diff_trajectory=self._get_traj_from_data(traj_diff),
            data_gen_strategy=strategy_used.value,
        )
        sample.resample_attempts = attempt
        return sample

    def _get_traj_dicts_for_rewind(self, ref_traj: dict) -> tuple[dict | Trajectory, dict] | None:
        """Get traj_sim and traj_diff for rewind strategy.

        Returns:
            Tuple of (traj_sim, traj_diff) where:
            - traj_sim = optimal trajectory from same task
            - traj_diff = rewound trajectory
            Returns None if either cannot be generated after retries.
            The main strategy loop will handle retries with different strategies.
        """
        max_retries = 3  # Number of retry attempts for sampling

        # Try to get optimal trajectory from same task for sim
        traj_sim = None
        for _ in range(max_retries):
            traj_sim = self._get_same_task_optimal(ref_traj)
            if traj_sim is not None:
                break

        # Try to get rewound trajectory for diff
        traj_diff = None
        for _ in range(max_retries):
            traj_diff = self._get_traj_from_data(ref_traj, subsample_strategy="subsample_rewind")
            if traj_diff is not None:
                break

        # Return both if successful, otherwise return None (main loop will handle retries)
        if traj_sim is not None and traj_diff is not None:
            return traj_sim, traj_diff

        return None

    def _get_traj_dicts_for_paired_human_robot(
        self, ref_traj: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Union[Dict[str, Any], Trajectory]]]:
        """Get traj_sim and traj_diff for paired human/robot strategy.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Tuple of (traj_sim, traj_diff) or None if not available. Both can be dict or Trajectory objects.
            traj_sim is the paired human/robot trajectory (opposite type, same task)
            traj_diff is a trajectory from a different task
        """
        max_retries = 3  # Number of retry attempts for sampling

        # Retry traj_sim separately
        traj_sim = None
        for _ in range(max_retries):
            traj_sim = self._get_paired_human_robot_traj(ref_traj)
            if traj_sim is not None:
                break

        # Retry traj_diff separately
        traj_diff = None
        for _ in range(max_retries):
            traj_diff = self._get_different_video_traj(ref_traj)
            if traj_diff is not None:
                break

        if traj_sim is not None and traj_diff is not None:
            return traj_sim, traj_diff

        return None

    def _get_traj_dicts_for_suboptimal(
        self, ref_traj: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Union[Dict[str, Any], Trajectory]]]:
        """Get traj_sim and traj_diff for suboptimal strategy.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Tuple of (traj_sim, traj_diff) or None if not available. Both can be dict or Trajectory objects.
            traj_sim is an optimal trajectory from same task
            traj_diff is a suboptimal trajectory from same task
        """
        max_retries = 3  # Number of retry attempts for sampling

        # Get optimal trajectory from same task for sim
        traj_sim = None
        for _ in range(max_retries):
            traj_sim = self._get_same_task_optimal(ref_traj)
            if traj_sim is not None:
                break

        # Get suboptimal trajectory from same task for diff
        traj_diff = None
        for _ in range(max_retries):
            traj_diff = self._get_same_task_suboptimal(ref_traj)
            if traj_diff is not None:
                break

        if traj_sim is not None and traj_diff is not None:
            return traj_sim, traj_diff

        return None
