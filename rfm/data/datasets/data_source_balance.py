import random
from collections import defaultdict
from typing import Any

from rfm.data.datasets.base import BaseDataset
from rfm.utils.logger import get_logger

logger = get_logger()


class DataSourceBalancedWrapper(BaseDataset):
    """Wrapper that applies data source balancing to any dataset.

    This wrapper can wrap RFMDataset, SingleFrameDataset, or any other dataset
    that inherits from BaseDataset. It overrides __getitem__ to sample based on
    data source weights while delegating all other functionality to the wrapped dataset.
    """

    def __init__(self, wrapped_dataset: BaseDataset, config, is_evaluation=False, **kwargs):
        """Initialize the wrapper.

        Args:
            wrapped_dataset: The dataset to wrap (e.g., RFMDataset or SingleFrameDataset)
            config: DataConfig with data_source_weights
            is_evaluation: Whether this is for evaluation
            **kwargs: Additional arguments (passed to BaseDataset.__init__)
        """
        # Initialize BaseDataset to get access to config and is_evaluation
        # We'll use the wrapped dataset's dataset attribute
        super().__init__(config, is_evaluation)

        self.wrapped_dataset = wrapped_dataset
        self.data_source_weights = config.data_source_weights or {}

        # Use the wrapped dataset's dataset attribute
        self.dataset = wrapped_dataset.dataset

        # Build source indices from the dataset
        logger.info("Building source indices for DataSourceBalancedWrapper...")
        self.source_indices = defaultdict(list)

        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        self._normalize_data_source_weights()

        logger.info(f"DataSourceBalancedWrapper initialized with {len(self.source_indices)} data sources")
        for source, indices in self.source_indices.items():
            weight = self.normalized_weights.get(source, 0.0)
            logger.info(f"  {source}: {len(indices)} trajectories (weight: {weight:.3f})")

        # Preserve data_len from wrapped dataset if it exists
        if hasattr(wrapped_dataset, "data_len"):
            self.data_len = wrapped_dataset.data_len
        else:
            self.data_len = len(self.dataset)

    def _normalize_data_source_weights(self):
        """Normalize data source weights across all available sources."""
        available_sources = list(self.source_indices.keys())

        if not available_sources:
            raise ValueError("DataSourceBalancedWrapper: no data sources found in dataset")

        # Get weights for available sources
        weights = {}
        total_weight = 0.0

        for source in available_sources:
            weight = self.data_source_weights.get(source, 1.0)
            weights[source] = weight
            total_weight += weight

        if total_weight <= 0:
            raise ValueError(
                f"DataSourceBalancedWrapper: total weight is {total_weight}, must be > 0. "
                f"Available sources: {available_sources}, weights: {self.data_source_weights}"
            )

        # Normalize weights
        self.normalized_weights = {source: weight / total_weight for source, weight in weights.items()}

    def __getitem__(self, idx):
        """Create a sample with balanced data source sampling.

        Overrides the wrapped dataset's __getitem__ to sample based on data source weights.
        """
        if not self.source_indices:
            raise ValueError("DataSourceBalancedWrapper has no indexed data sources.")

        max_source_attempts = max(len(self.source_indices) * 2, 1)
        if max_source_attempts <= 0:
            raise ValueError("DataSourceBalancedWrapper has no available data sources.")

        if not hasattr(self.wrapped_dataset, "_generate_sample_from_item"):
            raise ValueError(
                "DataSourceBalancedWrapper requires wrapped dataset to have _generate_sample_from_item method"
            )

        for _ in range(max_source_attempts):
            selected_source = self._select_weighted_source()
            source_indices = self.source_indices.get(selected_source)
            if not source_indices:
                continue

            # Select a trajectory index from the chosen source
            selected_traj_idx = random.choice(source_indices)

            # Get the item from the dataset
            item = self.dataset[selected_traj_idx]

            # Use the wrapped dataset's method to generate the sample
            return self.wrapped_dataset._generate_sample_from_item(item)

        raise ValueError("DataSourceBalancedWrapper: failed to generate a sample after exhausting data sources.")

    def _select_weighted_source(self) -> str:
        """Select a data source based on normalized weights."""
        available_sources = list(self.source_indices.keys())

        if len(available_sources) == 1:
            return available_sources[0]

        if not available_sources:
            raise ValueError("DataSourceBalancedWrapper has no available data sources.")

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

        # If we get here, no source was selected (all weights were 0 or negative)
        raise ValueError(
            f"DataSourceBalancedWrapper: failed to select a data source. "
            f"Available sources: {available_sources}, normalized_weights: {self.normalized_weights}"
        )

    def __len__(self):
        """Delegate to wrapped dataset's __len__ if available, otherwise use data_len."""
        if hasattr(self.wrapped_dataset, "__len__"):
            return len(self.wrapped_dataset)
        return self.data_len

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped dataset.

        This allows the wrapper to transparently forward all attributes and methods
        to the wrapped dataset, except for those explicitly defined in this class.
        """
        if name in [
            "wrapped_dataset",
            "source_indices",
            "normalized_weights",
            "data_source_weights",
            "data_len",
            "dataset",
            "config",
            "is_evaluation",
        ]:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.wrapped_dataset, name)
