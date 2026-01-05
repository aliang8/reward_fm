import random
from collections import defaultdict
from typing import Dict, List, Optional, Any

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.sim import SimSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.data.datasets.helpers import DataGenStrat
from rfm.data.dataset_category import (
    is_preference_only,
    is_suboptimal_fail_ds,
    is_paired_ds,
)
from rfm.utils.logger import get_logger

logger = get_logger()


class StrategyFirstDataset(BaseDataset):
    """
    Dataset that first selects sample type, then strategy, then picks a data source uniformly.

    This is different from RFMDataset which selects a trajectory first based on dataset iteration,
    and from StrategyBalancedDataset which selects sample type then data source (with optional weights).

    Sampling flow:
    1. Select sample type (preference/progress/similarity) based on sample_type_ratio
    2. Select strategy for that sample type based on strategy ratios
    3. Select data source uniformly from all available data sources
    4. Sample trajectory from selected data source and generate sample
    """

    def __init__(self, config, is_evaluation=False, max_samples=None, sampler_kwargs=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        self.pref_sampler = None
        self.progress_sampler = None
        self.similarity_sampler = None

        if sampler_kwargs is None:
            sampler_kwargs = {}

        base_sampler_kwargs = {
            "config": config,
            "dataset": self.dataset,
            "combined_indices": self._combined_indices,
            "dataset_success_cutoff_map": self.dataset_success_cutoff_map,
            "verbose": False,
            **sampler_kwargs,
        }

        if self.config.sample_type_ratio[0] > 0:
            self.pref_sampler = PrefSampler(is_evaluation=is_evaluation, **base_sampler_kwargs)
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = ProgressSampler(is_evaluation=is_evaluation, **base_sampler_kwargs)
        if self.config.sample_type_ratio[2] > 0:
            self.similarity_sampler = SimSampler(is_evaluation=is_evaluation, **base_sampler_kwargs)

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.data_len = len(self.dataset)

        # Build source indices for efficient uniform sampling
        self.source_indices = defaultdict(list)
        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        # Build set of tasks with optimal trajectories for efficient filtering
        self.tasks_with_optimal = set(self._combined_indices.get("optimal_by_task", {}).keys())
        
        # Build set of tasks with both optimal and suboptimal trajectories for SUBOPTIMAL strategy
        suboptimal_by_task = self._combined_indices.get("suboptimal_by_task", {})
        # Only include tasks that have non-empty suboptimal indices
        tasks_with_suboptimal = {task for task, indices in suboptimal_by_task.items() if indices}
        self.tasks_with_both = self.tasks_with_optimal & tasks_with_suboptimal
        
        # Build set of all indices from tasks with optimal trajectories for efficient filtering
        task_indices = self._combined_indices.get("task_indices", {})
        self.optimal_task_indices = set()
        for task in self.tasks_with_optimal:
            if task in task_indices:
                self.optimal_task_indices.update(task_indices[task])
        
        # Build set of all indices from tasks with both optimal and suboptimal trajectories
        self.tasks_with_both_indices = set()
        for task in self.tasks_with_both:
            if task in task_indices:
                self.tasks_with_both_indices.update(task_indices[task])

        logger.info(f"StrategyFirstDataset initialized with {len(self.dataset)} trajectories")
        logger.info(
            f"Sample type ratios: pref={self.sample_type_ratio[0]}, progress={self.sample_type_ratio[1]}, sim={self.sample_type_ratio[2]}"
        )
        logger.info(f"Available data sources: {list(self.source_indices.keys())}")
        logger.info(f"Tasks with optimal trajectories: {len(self.tasks_with_optimal)}")
        logger.info(f"Tasks with both optimal and suboptimal trajectories: {len(self.tasks_with_both)}")

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a sample by selecting sample type, then strategy, then data source uniformly."""
        logger.trace(f"[StrategyFirstDataset] __getitem__: Starting for idx={idx}")

        # Step 1: Select sample type based on ratios
        sample_type = self._select_sample_type()
        logger.trace(f"[StrategyFirstDataset] Selected sample type: {sample_type}")

        # Step 2: Select strategy for this sample type
        strategy = self._select_strategy(sample_type)
        if strategy is None:
            # Fallback: try to generate sample without specific strategy
            logger.trace(f"[StrategyFirstDataset] No strategy selected, using sampler default")
            return self._generate_without_specific_strategy(sample_type)

        logger.trace(
            f"[StrategyFirstDataset] Selected strategy: {strategy.value if hasattr(strategy, 'value') else strategy}"
        )

        # Step 3: Filter data sources based on strategy requirements
        filtered_sources = self._filter_data_sources_by_strategy(strategy)
        if not filtered_sources:
            logger.trace(
                f"[StrategyFirstDataset] No viable data sources for strategy {strategy.value if hasattr(strategy, 'value') else strategy}, retrying..."
            )
            # Retry by selecting a different strategy/sample type
            return self._generate_without_specific_strategy(sample_type)

        # Step 4: Select data source uniformly from filtered sources
        # Step 5: Filter indices based on strategy requirements
        # Step 6: Sample and generate
        max_attempts = 10  # Limit attempts to prevent infinite loops
        for attempt in range(max_attempts):
            selected_source = self._select_data_source_uniformly(filtered_sources)
            source_indices = self.source_indices.get(selected_source)

            if not source_indices:
                logger.trace(f"[StrategyFirstDataset] No indices for source {selected_source}, retrying...")
                continue

            # Filter indices based on strategy requirements
            filtered_indices = self._filter_indices_by_strategy(source_indices, selected_source, sample_type, strategy)
            if not filtered_indices:
                logger.trace(
                    f"[StrategyFirstDataset] No viable indices after strategy filtering for source {selected_source}, retrying..."
                )
                continue

            # Select a trajectory from filtered indices
            selected_traj_idx = random.choice(filtered_indices)
            item = self.dataset[selected_traj_idx]

            traj_id = item["id"]
            data_source = item["data_source"]
            quality_label = item["quality_label"]

            logger.trace(
                f"[StrategyFirstDataset] Attempt {attempt + 1}/{max_attempts}: "
                f"Selected traj ID={traj_id}, source={data_source}, quality={quality_label}, "
                f"sample_type={sample_type}, strategy={strategy.value if hasattr(strategy, 'value') else strategy}"
            )

            # Generate sample using the selected sampler with the preferred strategy
            sample = self._generate_sample_for_type(sample_type, item, preferred_strategy=strategy)
            if sample is not None:
                # Check if the generated sample matches our preferred strategy (if available)
                generated_strategy = getattr(sample, "data_gen_strategy", None)
                if generated_strategy:
                    logger.trace(
                        f"[StrategyFirstDataset] Generated sample with strategy {generated_strategy} "
                        f"(preferred: {strategy.value if hasattr(strategy, 'value') else strategy})"
                    )
                logger.trace(f"[StrategyFirstDataset] Successfully generated {sample_type} sample for ID={traj_id}")
                return self._set_resample_attempts(sample, attempt + 1)

            logger.trace(f"[StrategyFirstDataset] Sampler returned None for ID={traj_id}, retrying...")

        # All attempts failed
        logger.error(f"[StrategyFirstDataset] Failed to generate {sample_type} sample after {max_attempts} attempts")
        raise ValueError(f"Failed to generate {sample_type} sample after {max_attempts} attempts")

    def _select_sample_type(self) -> str:
        """Select a sample type based on sample_type_ratio."""
        available_types = []
        available_probs = []

        if self.sample_type_ratio[0] > 0 and self.pref_sampler is not None:
            available_types.append("pref")
            available_probs.append(self.sample_type_ratio[0])

        if self.sample_type_ratio[1] > 0 and self.progress_sampler is not None:
            available_types.append("progress")
            available_probs.append(self.sample_type_ratio[1])

        if self.sample_type_ratio[2] > 0 and self.similarity_sampler is not None:
            available_types.append("similarity")
            available_probs.append(self.sample_type_ratio[2])

        if not available_types:
            raise ValueError("No available sample types (all ratios are 0 or samplers are None)")

        # Normalize probabilities
        total_prob = sum(available_probs)
        normalized_probs = [p / total_prob for p in available_probs]

        # Select based on weighted random sampling
        prob = random.random()
        cumulative_prob = 0.0

        for sample_type, normalized_prob in zip(available_types, normalized_probs):
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return sample_type

        # Fallback (should not reach here)
        return available_types[-1]

    def _select_strategy(self, sample_type: str) -> Optional[DataGenStrat]:
        """Select a strategy for the given sample type based on strategy ratios."""
        strategies = []
        strategy_ratios = []

        if sample_type == "pref":
            strategy_ratios = self.config.preference_strategy_ratio
            # Map ratios to strategies: [rewind, suboptimal_same_task, different_task, reverse_progress]
            if len(strategy_ratios) > 0 and strategy_ratios[0] > 0:
                strategies.append((DataGenStrat.REWIND, strategy_ratios[0]))
            if len(strategy_ratios) > 1 and strategy_ratios[1] > 0:
                strategies.append((DataGenStrat.SUBOPTIMAL, strategy_ratios[1]))
            if len(strategy_ratios) > 2 and strategy_ratios[2] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK, strategy_ratios[2]))
            if len(strategy_ratios) > 3 and strategy_ratios[3] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, strategy_ratios[3]))

        elif sample_type == "progress":
            strategy_ratios = self.config.progress_strategy_ratio
            # Map ratios to strategies: [different_task_instruction, forward_progress, reverse_progress, rewind]
            if len(strategy_ratios) > 0 and strategy_ratios[0] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK_INSTRUCTION, strategy_ratios[0]))
            if len(strategy_ratios) > 1 and strategy_ratios[1] > 0:
                strategies.append((DataGenStrat.FORWARD_PROGRESS, strategy_ratios[1]))
            if len(strategy_ratios) > 2 and strategy_ratios[2] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, strategy_ratios[2]))
            if len(strategy_ratios) > 3 and strategy_ratios[3] > 0:
                strategies.append((DataGenStrat.REWIND, strategy_ratios[3]))

        elif sample_type == "similarity":
            strategy_ratios = self.config.similarity_strategy_ratio
            # Map ratios to strategies: [rewind, suboptimal_same_task, paired_human_robot]
            if len(strategy_ratios) > 0 and strategy_ratios[0] > 0:
                strategies.append((DataGenStrat.REWIND, strategy_ratios[0]))
            if len(strategy_ratios) > 1 and strategy_ratios[1] > 0:
                strategies.append((DataGenStrat.SUBOPTIMAL, strategy_ratios[1]))
            if len(strategy_ratios) > 2 and strategy_ratios[2] > 0:
                strategies.append((DataGenStrat.PAIRED_HUMAN_ROBOT, strategy_ratios[2]))

        if not strategies:
            return None

        # Normalize probabilities
        total_prob = sum(prob for _, prob in strategies)
        if total_prob == 0:
            return None

        normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

        # Select based on weighted random sampling
        prob = random.random()
        cumulative_prob = 0.0

        for strategy, normalized_prob in normalized_strategies:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return strategy

        # Fallback
        return strategies[0][0]

    def _select_data_source_uniformly(self, allowed_sources: Optional[List[str]] = None) -> str:
        """Select a data source uniformly from allowed data sources.

        Args:
            allowed_sources: Optional list of allowed data source names. If None, uses all available sources.

        Returns:
            Selected data source name
        """
        if allowed_sources is None:
            available_sources = list(self.source_indices.keys())
        else:
            # Filter to only include sources that exist in our source_indices
            available_sources = [source for source in allowed_sources if source in self.source_indices]

        if not available_sources:
            raise ValueError("No available data sources")

        return random.choice(available_sources)

    def _filter_data_sources_by_strategy(self, strategy: Optional[DataGenStrat]) -> List[str]:
        """Filter data sources based on strategy requirements.

        Args:
            strategy: The selected strategy

        Returns:
            List of viable data source names for the strategy
        """
        all_sources = list(self.source_indices.keys())

        if strategy is None:
            return all_sources

        # Filter based on strategy requirements
        if strategy == DataGenStrat.SUBOPTIMAL:
            # SUBOPTIMAL strategy needs data sources with suboptimal/failure trajectories
            filtered = [source for source in all_sources if is_suboptimal_fail_ds(source)]
            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(all_sources)} data sources for SUBOPTIMAL strategy"
            )
            return filtered if filtered else all_sources

        elif strategy == DataGenStrat.PAIRED_HUMAN_ROBOT:
            # PAIRED_HUMAN_ROBOT strategy needs paired data sources
            filtered = [source for source in all_sources if is_paired_ds(source)]
            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(all_sources)} data sources for PAIRED_HUMAN_ROBOT strategy"
            )
            return filtered if filtered else all_sources

        # Other strategies (REWIND, DIFFERENT_TASK, REVERSE_PROGRESS, etc.) can work with any data source
        return all_sources

    def _filter_indices_by_strategy(
        self, indices: List[int], data_source: str, sample_type: str, strategy: Optional[DataGenStrat]
    ) -> List[int]:
        """Filter indices based on strategy requirements.

        For SUBOPTIMAL strategy (preference or similarity), filters to only include indices from tasks
        that have optimal trajectories (unless RoboArena).

        Args:
            indices: List of trajectory indices to filter
            data_source: The data source name
            sample_type: The sample type (pref/progress/similarity)
            strategy: The selected strategy

        Returns:
            Filtered list of viable indices for the strategy
        """
        if strategy is None:
            return indices

        # Check if this is RoboArena (skip task filtering for RoboArena)
        is_roboarena = data_source and "roboarena" in str(data_source).lower()

        # For SUBOPTIMAL strategy (preference or similarity), filter to tasks with both optimal and suboptimal trajectories
        if strategy == DataGenStrat.SUBOPTIMAL and sample_type in ["pref", "similarity"]:
            if is_roboarena:
                # RoboArena uses partial_success logic, don't filter by task requirements
                return indices

            if not self.tasks_with_both:
                # No tasks with both optimal and suboptimal trajectories, return empty list
                logger.trace(
                    f"[StrategyFirstDataset] No tasks with both optimal and suboptimal trajectories available for SUBOPTIMAL strategy"
                )
                return []

            # Use pre-computed tasks_with_both_indices and intersect with our current indices
            indices_set = set(indices)
            filtered = self.tasks_with_both_indices & indices_set

            if not filtered:
                logger.trace(
                    f"[StrategyFirstDataset] No viable indices after filtering for SUBOPTIMAL strategy"
                )
                return []

            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(indices)} indices for SUBOPTIMAL strategy "
                f"(keeping only tasks with both optimal and suboptimal trajectories)"
            )
            return list(filtered)

        # Other strategies don't require task-level filtering
        return indices

    def _generate_sample_for_type(
        self, sample_type: str, item: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None
    ):
        """Generate a sample using the appropriate sampler for the sample type.
        
        Args:
            sample_type: The sample type (pref/progress/similarity)
            item: The trajectory item
            preferred_strategy: Optional strategy to use (if None, sampler will select its own)
        """
        data_source = item["data_source"]
        quality_label = item["quality_label"]

        # Get the appropriate sampler
        if sample_type == "pref":
            sampler = self.pref_sampler
        elif sample_type == "progress":
            sampler = self.progress_sampler
        elif sample_type == "similarity":
            sampler = self.similarity_sampler
        else:
            return None

        if sampler is None:
            return None

        # Handle non-successful trajectories: force preference-only
        if quality_label != "successful" and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(f"[StrategyFirstDataset] Non-successful quality detected, switching to preference sampler")
                return self.pref_sampler._generate_sample(item, preferred_strategy=preferred_strategy)
            else:
                return None

        # Handle preference-only data sources
        if is_preference_only(data_source) and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(
                    f"[StrategyFirstDataset] Preference-only data source detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item, preferred_strategy=preferred_strategy)
            else:
                return None

        # Generate sample using the selected sampler with preferred strategy
        return sampler._generate_sample(item, preferred_strategy=preferred_strategy)

    def _generate_without_specific_strategy(self, sample_type: str):
        """Fallback method to generate sample without specific strategy selection."""
        max_attempts = 10
        for attempt in range(max_attempts):
            selected_source = self._select_data_source_uniformly()
            source_indices = self.source_indices.get(selected_source)

            if not source_indices:
                continue

            selected_traj_idx = random.choice(source_indices)
            item = self.dataset[selected_traj_idx]

            sample = self._generate_sample_for_type(sample_type, item)
            if sample is not None:
                return self._set_resample_attempts(sample, attempt + 1)

        raise ValueError(f"Failed to generate {sample_type} sample after {max_attempts} attempts")

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats
