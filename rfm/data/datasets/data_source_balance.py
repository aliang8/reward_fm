import random
from collections import defaultdict

from rfm.data.datasets.rfm_data import RFMDataset
from rfm.utils.logger import get_logger

logger = get_logger()


class BalancedRFMDataset(RFMDataset):
    """Dataset that extends RFMDataset with configurable sampling weights per data source."""

    def __init__(self, config, is_evaluation=False, max_samples=None, **kwargs):
        super().__init__(config, is_evaluation, max_samples, **kwargs)

        self.data_source_weights = config.data_source_weights
        self.data_len = 10000000

        logger.info("Building source indices for BalancedRFMDataset...")
        self.source_indices = defaultdict(list)

        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        self._normalize_data_source_weights()
        
        logger.info(f"BalancedRFMDataset initialized with {len(self.source_indices)} data sources")
        for source, indices in self.source_indices.items():
            weight = self.normalized_weights.get(source, 0.0)
            logger.info(f"  {source}: {len(indices)} trajectories (weight: {weight:.3f})")

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

    def __getitem__(self, idx):
        """Create a sample with balanced data source sampling and shared sampler logic."""
        if not self.source_indices:
            raise ValueError("BalancedRFMDataset has no indexed data sources.")

        max_source_attempts = max(len(self.source_indices) * 2, 1)
        if max_source_attempts <= 0:
            raise ValueError("BalancedRFMDataset has no available data sources.")

        for _ in range(max_source_attempts):
            selected_source = self._select_weighted_source()
            source_indices = self.source_indices.get(selected_source)
            if not source_indices:
                continue

            selected_traj_idx = random.choice(source_indices)
            item = self.dataset[selected_traj_idx]

            try:
                return self._generate_sample_from_item(item)
            except ValueError:
                # Try another source if sampling for this item failed
                continue

        raise ValueError("BalancedRFMDataset: failed to generate a sample after exhausting data sources.")

    def _select_weighted_source(self) -> str:
        """Select a data source based on normalized weights."""
        available_sources = list(self.source_indices.keys())

        if len(available_sources) == 1:
            return available_sources[0]

        # Select based on weighted random sampling
        prob = random.random()
        cumulative_prob = 0.0

        for source in available_sources:
            weight = self.normalized_weights.get(source, 0.0)
            if weight <= 0:
                continue
            cumulative_prob += weight
            if prob <= cumulative_prob:
                return source

        # Fallback (should not reach here)
        return available_sources[-1]
