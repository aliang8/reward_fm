import random
from collections import defaultdict

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.sim import SimSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.data.dataset_category import is_preference_only
from rfm.utils.distributed import rank_0_print
from rfm.utils.logger import get_logger

logger = get_logger()


class StrategyBalancedDataset(BaseDataset):
    """
    Dataset that first selects a sample type (preference, progress, similarity) based on ratios,
    then selects a dataset source and trajectory to construct the sample from.

    This is different from RFMDataset which selects a trajectory first, then selects a sample type.
    """

    def __init__(self, config, is_evaluation=False, max_samples=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        self.pref_sampler = None
        self.progress_sampler = None
        self.similarity_sampler = None

        if self.config.sample_type_ratio[0] > 0:
            self.pref_sampler = PrefSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = ProgressSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )
        if self.config.sample_type_ratio[2] > 0:
            self.similarity_sampler = SimSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.data_len = len(self.dataset)

        # Build source indices for efficient sampling
        self.source_indices = defaultdict(list)
        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        # Handle data source weights if provided
        self.data_source_weights = getattr(config, "data_source_weights", None)
        if self.data_source_weights:
            self._normalize_data_source_weights()
        else:
            self.normalized_weights = None

        logger.info(f"StrategyBalancedDataset initialized with {len(self.dataset)} trajectories")
        logger.info(
            f"Sample type ratios: pref={self.sample_type_ratio[0]}, progress={self.sample_type_ratio[1]}, sim={self.sample_type_ratio[2]}"
        )
        logger.info(f"Available data sources: {list(self.source_indices.keys())}")

    def _normalize_data_source_weights(self):
        """Normalize data source weights across all available sources."""
        available_sources = list(self.source_indices.keys())

        if not available_sources:
            self.normalized_weights = {}
            return

        # Get weights for available sources
        weights = {}
        total_weight = 0.0

        for source in available_sources:
            weight = self.data_source_weights.get(source, 1.0)
            weights[source] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            self.normalized_weights = {source: weight / total_weight for source, weight in weights.items()}
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(available_sources)
            self.normalized_weights = {source: equal_weight for source in available_sources}

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a sample by first selecting sample type, then dataset source and trajectory."""
        logger.trace(f"[StrategyBalancedDataset] __getitem__: Starting for idx={idx}")

        # Step 1: Select sample type based on ratios
        sample_type = self._select_sample_type()
        logger.trace(f"[StrategyBalancedDataset] __getitem__: Selected sample type: {sample_type}")

        # Step 2: Select appropriate sampler
        sampler = self._get_sampler_for_type(sample_type)
        if sampler is None:
            raise ValueError(f"No sampler available for sample type: {sample_type}")

        # Step 3: Select dataset source and trajectory
        max_attempts = 20  # Limit attempts to prevent infinite loops
        for attempt in range(max_attempts):
            # Select a dataset source
            selected_source = self._select_data_source()
            source_indices = self.source_indices.get(selected_source)

            if not source_indices:
                logger.trace(f"[StrategyBalancedDataset] No indices for source {selected_source}, retrying...")
                continue

            # Select a trajectory from this source
            selected_traj_idx = random.choice(source_indices)
            item = self.dataset[selected_traj_idx]

            traj_id = item["id"]
            data_source = item["data_source"]
            quality_label = item["quality_label"]

            logger.trace(
                f"[StrategyBalancedDataset] Attempt {attempt + 1}/{max_attempts}: "
                f"Selected traj ID={traj_id}, source={data_source}, quality={quality_label} for {sample_type}"
            )

            # Step 4: Generate sample using the selected sampler
            try:
                sample = self._generate_sample_for_type(sampler, sample_type, item)
                if sample is not None:
                    logger.trace(
                        f"[StrategyBalancedDataset] Successfully generated {sample_type} sample for ID={traj_id}"
                    )
                    return self._set_resample_attempts(sample, attempt + 1)
                else:
                    logger.trace(f"[StrategyBalancedDataset] Sampler returned None for ID={traj_id}, retrying...")
            except ValueError as e:
                logger.trace(f"[StrategyBalancedDataset] ValueError for ID={traj_id}: {e}, retrying...")
                continue

        # All attempts failed
        logger.error(f"[StrategyBalancedDataset] Failed to generate {sample_type} sample after {max_attempts} attempts")
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

    def _get_sampler_for_type(self, sample_type: str):
        """Get the sampler for a given sample type."""
        if sample_type == "pref":
            return self.pref_sampler
        elif sample_type == "progress":
            return self.progress_sampler
        elif sample_type == "similarity":
            return self.similarity_sampler
        else:
            return None

    def _select_data_source(self) -> str:
        """Select a data source, optionally using weights."""
        available_sources = list(self.source_indices.keys())

        if not available_sources:
            raise ValueError("No available data sources")

        if len(available_sources) == 1:
            return available_sources[0]

        # Use weighted selection if weights are provided
        if self.normalized_weights:
            prob = random.random()
            cumulative_prob = 0.0

            for source in available_sources:
                weight = self.normalized_weights.get(source, 0.0)
                if weight <= 0:
                    continue
                cumulative_prob += weight
                if prob <= cumulative_prob:
                    return source

            # Fallback
            return available_sources[-1]
        else:
            # Uniform selection
            return random.choice(available_sources)

    def _generate_sample_for_type(self, sampler, sample_type: str, item: dict):
        """Generate a sample using the appropriate sampler and handle special cases."""
        data_source = item["data_source"]
        quality_label = item["quality_label"]

        # Handle non-successful trajectories: force preference-only
        if quality_label != "successful" and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(
                    f"[StrategyBalancedDataset] Non-successful quality detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item)
            else:
                # Can't use preference, try to use progress as fallback
                if sample_type == "progress" and self.progress_sampler is not None:
                    return self.progress_sampler._generate_sample(item)
                # Otherwise, return None to trigger retry
                return None

        # Handle preference-only data sources
        if is_preference_only(data_source) and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(
                    f"[StrategyBalancedDataset] Preference-only data source detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item)
            else:
                # Can't use preference, return None to trigger retry
                return None

        # Generate sample using the selected sampler
        return sampler._generate_sample(item)

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats
