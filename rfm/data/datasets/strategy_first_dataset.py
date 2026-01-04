import random
from collections import defaultdict
from typing import Dict, List, Optional, Any

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.sim import SimSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.data.datasets.helpers import DataGenStrat
from rfm.data.dataset_category import is_preference_only
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

        logger.info(f"StrategyFirstDataset initialized with {len(self.dataset)} trajectories")
        logger.info(
            f"Sample type ratios: pref={self.sample_type_ratio[0]}, progress={self.sample_type_ratio[1]}, sim={self.sample_type_ratio[2]}"
        )
        logger.info(f"Available data sources: {list(self.source_indices.keys())}")

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

        logger.trace(f"[StrategyFirstDataset] Selected strategy: {strategy.value if hasattr(strategy, 'value') else strategy}")

        # Step 3: Select data source uniformly
        # Step 4: Sample and generate
        max_attempts = 10  # Limit attempts to prevent infinite loops
        for attempt in range(max_attempts):
            selected_source = self._select_data_source_uniformly()
            source_indices = self.source_indices.get(selected_source)

            if not source_indices:
                logger.trace(f"[StrategyFirstDataset] No indices for source {selected_source}, retrying...")
                continue

            # Select a trajectory from this source
            selected_traj_idx = random.choice(source_indices)
            item = self.dataset[selected_traj_idx]

            traj_id = item["id"]
            data_source = item["data_source"]
            quality_label = item["quality_label"]

            logger.trace(
                f"[StrategyFirstDataset] Attempt {attempt + 1}/{max_attempts}: "
                f"Selected traj ID={traj_id}, source={data_source}, quality={quality_label}, "
                f"sample_type={sample_type}, strategy={strategy.value if hasattr(strategy, 'value') else strategy}"
            )

            # Generate sample using the selected sampler
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

    def _select_data_source_uniformly(self) -> str:
        """Select a data source uniformly from all available data sources."""
        available_sources = list(self.source_indices.keys())

        if not available_sources:
            raise ValueError("No available data sources")

        return random.choice(available_sources)

    def _generate_sample_for_type(
        self, sample_type: str, item: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None
    ):
        """Generate a sample using the appropriate sampler for the sample type."""
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
                logger.trace(
                    f"[StrategyFirstDataset] Non-successful quality detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item)
            else:
                return None

        # Handle preference-only data sources
        if is_preference_only(data_source) and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(
                    f"[StrategyFirstDataset] Preference-only data source detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item)
            else:
                return None

        # Generate sample using the selected sampler
        return sampler._generate_sample(item)

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

            sample = self._generate_sample_for_type(sample_type, item, preferred_strategy=None)
            if sample is not None:
                return self._set_resample_attempts(sample, attempt + 1)

        raise ValueError(f"Failed to generate {sample_type} sample after {max_attempts} attempts")

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats

