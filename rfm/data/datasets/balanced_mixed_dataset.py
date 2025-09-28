import random
from collections import defaultdict

from .mixed_dataset import MixedDataset
from rfm.utils.distributed import rank_0_print


class BalancedMixedDataset(MixedDataset):
    """Dataset that extends MixedDataset with configurable sampling weights per data source."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, max_samples, batch_size, **kwargs)

        self.data_source_weights = config.data_source_weights
        self.data_len = 100000

        rank_0_print("Building source indices for BalancedMixedDataset...")
        self.source_indices = {}
        for i, traj in enumerate(self.dataset):
            source = traj["data_source"]
            if source not in self.source_indices:
                self.source_indices[source] = []
            self.source_indices[source].append(i)

        # Normalize data source weights
        self._normalize_data_source_weights()

        if self.verbose:
            rank_0_print(f"BalancedMixedDataset initialized with {len(self.source_indices)} data sources")
            for source, indices in self.source_indices.items():
                weight = self.normalized_weights.get(source, 0.0)
                rank_0_print(f"  {source}: {len(indices)} trajectories (weight: {weight:.3f})")

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
            weight = self.data_source_weights.get(source, 1.0)  # Default weight of 1.0
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
        """Create a sample with balanced data source sampling and configured sample type ratios."""
        # Available dataset types with their probabilities
        datasets = [
            ("pref", self.sample_type_ratio[0], self.pref_dataset),
            ("progress", self.sample_type_ratio[1], self.progress_dataset),
            ("similarity", self.sample_type_ratio[2], self.similarity_dataset),
        ]

        # Remove datasets with zero probability
        available_datasets = [(name, prob, dataset) for name, prob, dataset in datasets if prob > 0]

        # Normalize probabilities
        total_prob = sum(prob for _, prob, _ in available_datasets)
        normalized_datasets = [(name, prob / total_prob, dataset) for name, prob, dataset in available_datasets]

        # Select dataset based on normalized probabilities
        prob = random.random()
        cumulative_prob = 0.0

        for name, normalized_prob, dataset in normalized_datasets:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return self._get_balanced_sample(name)

        import ipdb

        ipdb.set_trace()

    def _get_balanced_sample(self, sample_type: str):
        """Get a sample of the specified type with weighted data source sampling."""
        # Select data source based on unified normalized weights
        selected_source = self._select_weighted_source()

        # Get available trajectory indices for this source
        source_indices = self.source_indices[selected_source]

        # Select trajectory index randomly within the source
        selected_traj_idx = random.choice(source_indices)

        # Generate sample using the appropriate dataset
        if sample_type == "pref":
            return self.pref_dataset[selected_traj_idx]
        elif sample_type == "progress":
            return self.progress_dataset[selected_traj_idx]
        elif sample_type == "similarity":
            return self.similarity_dataset[selected_traj_idx]

    def _select_weighted_source(self) -> str:
        """Select a data source based on normalized weights."""
        available_sources = list(self.source_indices.keys())

        if len(available_sources) == 1:
            return available_sources[0]

        # Select based on weighted random sampling
        prob = random.random()
        cumulative_prob = 0.0

        for source in available_sources:
            weight = self.normalized_weights[source]
            cumulative_prob += weight
            if prob <= cumulative_prob:
                return source

        import ipdb

        ipdb.set_trace()


def test():
    """Test the BalancedMixedDataset with generated samples."""
    # Create a mock config for testing
    from dataclasses import dataclass

    @dataclass
    class MockDataConfig:
        train_datasets: list[str] = None
        train_subsets: list[str] = None
        eval_datasets: list[str] = None
        eval_subsets: list[str] = None
        sample_type_ratio: list[float] = None
        shuffle: bool = True
        seed: int = 42
        num_proc: int = 4
        max_frames: int = 8
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: list[float] = None
        progress_strategy_ratio: list[float] = None
        dataset_preference_ratio: float = None
        data_source_weights: dict[str, float] = None
        load_embeddings: bool = False

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["aliangdw/metaworld_rfm", "abraranwar/libero_rfm", "ykorkmaz/libero_failure_rfm"],
        train_subsets=[["metaworld"], ["libero256_10"], ["libero_10_failure"]],
        sample_type_ratio=[0, 1, 0],  # pref, progress, similarity
        preference_strategy_ratio=[0.8, 0.1, 0.1, 0.0],
        progress_strategy_ratio=[0.8, 0.1, 0.1],
        dataset_preference_ratio=0.8,
        data_source_weights={
            "metaworld": 5,
            "libero256_10": 1,
            "libero_10_failure": 1,
        },
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,
        force_reprocess=False,
        model_type="default",
        load_embeddings=True,
    )

    MockConfig(data=mock_data_config, debug=False)

    # Create balanced dataset generator
    generator = BalancedMixedDataset(config=mock_data_config)

    # Test the dataset
    rank_0_print("Testing BalancedMixedDataset...")

    sample_type_counts = {"pref": 0, "progress": 0, "similarity": 0}
    source_counts = defaultdict(int)

    for i in range(100):
        sample = generator[i]

        # Determine sample type
        if hasattr(sample, "chosen_trajectory"):
            sample_type_counts["pref"] += 1
            source_counts[sample.chosen_trajectory.data_source] += 1
        elif hasattr(sample, "trajectory"):
            if hasattr(sample, "reference_trajectory"):
                sample_type_counts["similarity"] += 1
                source_counts[sample.trajectory.data_source] += 1
            else:
                sample_type_counts["progress"] += 1
                source_counts[sample.trajectory.data_source] += 1

    rank_0_print(f"Sample type distribution: {sample_type_counts}")
    rank_0_print(f"Data source distribution: {dict(source_counts)}")
    rank_0_print(f"Total samples tested: {sum(sample_type_counts.values())}")


if __name__ == "__main__":
    test()
