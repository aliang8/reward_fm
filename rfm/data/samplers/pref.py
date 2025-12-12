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
from rfm.utils.logger import get_logger, rank_0_info, trace
from rfm.utils.timer import timer

logger = get_logger()


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
        self._has_suboptimal = (
            any(len(indices) > 0 for indices in self.suboptimal_by_task.values()) if self.suboptimal_by_task else False
        )
        rank_0_info(f"[PREF SAMPLER] Has suboptimal: {self._has_suboptimal}")

        # Initialize preference dataset
        self._load_preference_dataset()

    def _generate_sample(self, item: dict):
        """Generate a preference sample from an item.

        If the item has a non-successful quality label, it will be used as the rejected
        trajectory and an optimal trajectory from the same task will be found as the chosen one.
        Otherwise, normal preference sampling logic is used.
        """
        quality_label = item["quality_label"]
        is_roboarena = "roboarena" in str(item.get("data_source", "")).lower()

        # Handle non-successful trajectories: use as rejected, find optimal from same task as chosen
        # skip this for RoboArena trajectories which we will handle with partial success
        if quality_label != "successful" and not is_roboarena:
            traj_id = item["id"]
            task_name = item["task"]

            logger.trace(
                f"[PREF SAMPLER] Non-successful quality detected for ID={traj_id}, using as rejected trajectory, task={task_name}"
            )

            # Find optimal trajectories from the same task
            same_task_optimal_indices = self.optimal_by_task.get(task_name, [])

            if not same_task_optimal_indices:
                logger.trace(
                    f"[PREF SAMPLER] No optimal trajectories found for task '{task_name}', falling through to normal sampling"
                )
                return self._create_pref_sample(item)

            # Select a random optimal trajectory from the same task as chosen
            chosen_idx = random.choice(same_task_optimal_indices)
            chosen_traj_dict = self.dataset[chosen_idx]

            # Create trajectories using the base sampler's method
            chosen_trajectory = self._get_traj_from_data(chosen_traj_dict)
            rejected_trajectory = self._get_traj_from_data(item)

            # Set rejected trajectory progress to 0 (as per suboptimal strategy)
            rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)

            # Create preference sample with suboptimal strategy
            sample = PreferenceSample(
                chosen_trajectory=chosen_trajectory,
                rejected_trajectory=rejected_trajectory,
                data_gen_strategy=DataGenStrat.SUBOPTIMAL.value,
            )

            logger.trace(
                f"[PREF SAMPLER] Created preference sample for non-successful traj ID={traj_id} with optimal traj from same task"
            )
            return sample

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
        rank_0_info("[PREF SAMPLER] No preference dataset provided, will use random sampling for preferences")
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
        # Log when preference sampler is called
        traj_id = chosen_traj["id"] if chosen_traj is not None else "sampling_new"
        logger.trace(f"[PREF SAMPLER] Creating preference sample for trajectory ID: {traj_id}")

        # Use provided chosen trajectory if given; otherwise sample one
        if chosen_traj is None:
            # Use preprocessed chosen trajectories from index maps
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

            # Get a random task and chosen trajectory from it
            task_name = random.choice(list(valid_tasks.keys()))
            optimal_indices = valid_tasks[task_name]

            # Double-check that we have valid indices (should always be true now)
            if not optimal_indices:
                return None

            chosen_idx = random.choice(optimal_indices)
            chosen_traj = self.dataset[chosen_idx]

        # Initialize variables for strategy selection
        rejected_traj = None
        strategy_used = None

        # Check if this is a RoboArena trajectory (has partial_success and data_source contains "roboarena")
        is_roboarena = False
        data_source = chosen_traj.get("data_source", "")
        partial_success = chosen_traj.get("partial_success")
        if partial_success is not None and data_source and "roboarena" in str(data_source).lower():
            is_roboarena = True
            logger.trace(
                f"[PREF SAMPLER] RoboArena trajectory detected (ID: {chosen_traj.get('id', 'unknown')}, partial_success: {partial_success})"
            )

        # Strategy selection with rebalancing on failure
        strategies = []
        # # For RoboArena, always use partial_success strategy only
        # if is_roboarena:
        #     strategies.append((DataGenStrat.ROBOARENA_PARTIAL_SUCCESS, 1.0))
        # else:

        # Add other strategies if not RoboArena
        if self.preference_strategy_ratio[0] > 0:
            strategies.append((DataGenStrat.REWOUND, self.preference_strategy_ratio[0]))
        if self._has_suboptimal and self.preference_strategy_ratio[1] > 0:
            strategies.append((DataGenStrat.SUBOPTIMAL, self.preference_strategy_ratio[1]))
        if self.preference_strategy_ratio[2] > 0:
            strategies.append((DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]))

        if is_roboarena:
            strategies.append((DataGenStrat.ROBOARENA_PARTIAL_SUCCESS, 10.0))
            # remove suboptimal strategy
            strategies = [(strat, prob) for strat, prob in strategies if strat != DataGenStrat.SUBOPTIMAL]

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

            # Log strategy attempt
            logger.trace(
                f"[PREF SAMPLER] Attempt {attempt}/{max_attempts}: Trying strategy {selected_strategy.value if selected_strategy else 'None'}"
            )

            # Execute selected strategy with retry logic
            max_retries = 3  # Number of retry attempts for sampling

            if selected_strategy == DataGenStrat.ROBOARENA_PARTIAL_SUCCESS:
                rejected_traj = None
                for _ in range(max_retries):
                    different_traj = self._get_different_partial_success_traj(chosen_traj)
                    if different_traj is not None:
                        # If the returned trajectory has higher partial_success, swap them
                        # so the higher one becomes chosen and the lower one becomes rejected
                        chosen_partial_success = chosen_traj.get("partial_success")
                        different_partial_success = different_traj.get("partial_success")
                        if different_partial_success is not None and chosen_partial_success is not None:
                            if different_partial_success > chosen_partial_success:
                                # Swap: higher becomes chosen, original chosen becomes rejected
                                logger.trace(
                                    f"[PREF SAMPLER] Swapping trajectories: found higher partial_success "
                                    f"({different_partial_success} > {chosen_partial_success}), making higher trajectory chosen"
                                )
                                rejected_traj = chosen_traj
                                chosen_traj = different_traj
                            else:
                                # Lower becomes rejected
                                rejected_traj = different_traj
                        else:
                            rejected_traj = different_traj
                        break
            elif selected_strategy == DataGenStrat.REWOUND:
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
                logger.trace(f"[PREF SAMPLER] Strategy {selected_strategy.value} succeeded on attempt {attempt}")
            else:
                # Strategy failed - increment attempt count
                strategy_attempt_counts[selected_strategy] = strategy_attempt_counts.get(selected_strategy, 0) + 1
                failed_count = strategy_attempt_counts[selected_strategy]

                logger.trace(
                    f"[PREF SAMPLER] Strategy {selected_strategy.value} failed (failure count: {failed_count}/{max_strategy_attempts})"
                )

                # Only remove strategy if it has failed max_strategy_attempts times
                if strategy_attempt_counts[selected_strategy] >= max_strategy_attempts:
                    logger.trace(
                        f"[PREF SAMPLER] Removing strategy {selected_strategy.value} after {max_strategy_attempts} consecutive failures"
                    )
                    strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                    continue

        # If we still don't have a sample after all attempts, return None
        if rejected_traj is None:
            logger.trace(
                f"[PREF SAMPLER] Failed to generate preference sample after {max_attempts} attempts - all strategies exhausted"
            )
            return None

        chosen_trajectory = self._get_traj_from_data(chosen_traj)
        rejected_trajectory = self._get_traj_from_data(rejected_traj)

        # If our strategy is different task or suboptimal, make sure the rejected trajectory has 0 progress
        # For RoboArena partial_success, keep the original progress (chosen has higher partial_success, rejected has lower)
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
