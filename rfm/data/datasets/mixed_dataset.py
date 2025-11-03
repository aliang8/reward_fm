import random

from .base import RFMBaseDataset
from .pref import PrefDataset
from .sim import SimilarityDataset
from .progress import ProgressDataset
from rfm.utils.distributed import rank_0_print


class MixedDataset(RFMBaseDataset):
    """Dataset that combines preference, similarity, and progress generation."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        # Initialize the individual datasets
        self.pref_dataset = PrefDataset(config, is_evaluation, verbose=False, **kwargs)
        self.progress_dataset = ProgressDataset(config, is_evaluation, verbose=False, **kwargs)
        self.similarity_dataset = SimilarityDataset(config, is_evaluation, verbose=False, **kwargs)

        # Set the ratio for sampling between preference, similarity, and progress
        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.data_len = max(len(self.pref_dataset), len(self.similarity_dataset), len(self.progress_dataset))

    def __len__(self):
        if self.max_samples is None:
            return max(self.data_len, self.batch_size)
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a sample based on the configured ratios using normalized and rebalanced selection.

        Uses the same normalization and rebalancing approach as PrefDataset and ProgressDataset
        to handle cases where some sample types have zero probability or datasets are unavailable.
        """
        idx = idx % self.data_len

        # Preference-only override by data_source using raw filtered dataset entry
        pref_only = getattr(self.config, "pref_only_datasets", []) or []
        try:
            base_entry = self.filtered_dataset[idx]
            data_source = base_entry.get("data_source", None)
        except Exception:
            data_source = None
        if data_source in pref_only:
            pref_idx = idx % max(1, len(self.pref_dataset))
            return self.pref_dataset[pref_idx]

        # Available dataset types with their probabilities
        datasets = [
            ("pref", self.sample_type_ratio[0], self.pref_dataset),
            ("progress", self.sample_type_ratio[1], self.progress_dataset),
            ("similarity", self.sample_type_ratio[2], self.similarity_dataset),
        ]

        # Remove datasets with zero probability
        available_datasets = [(name, prob, dataset) for name, prob, dataset in datasets if prob > 0]

        # Fallback to progress dataset if no datasets have positive probability
        if not available_datasets:
            return self.progress_dataset[idx]

        # Normalize probabilities
        total_prob = sum(prob for _, prob, _ in available_datasets)
        normalized_datasets = [(name, prob / total_prob, dataset) for name, prob, dataset in available_datasets]

        # Select dataset based on normalized probabilities
        prob = random.random()
        cumulative_prob = 0.0

        for name, normalized_prob, dataset in normalized_datasets:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return dataset[idx]

        # Final fallback (should not reach here, but safety net)
        return self.progress_dataset[idx]


def test():
    """Test the BatchCollator with generated samples."""
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
        max_frames: int = 8  # Use 8 frames for testing the new subsampling logic
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: list[float] = None

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["jesbu1/oxe_rfm"],
        train_subsets=["oxe_bridge_v2"],
        sample_type_ratio=[1.0, 0.0, 0.0],
        preference_strategy_ratio=[0.8, 0.1, 0.1, 0.0],
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,  # Use 8 frames for testing the new subsampling logic
        force_reprocess=False,
        model_type="default",
    )

    MockConfig(data=mock_data_config, debug=False)

    # Create data generator with mock config
    generator = MixedDataset(config=mock_data_config)

    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")

    preference_count = 0
    similarity_count = 0

    for i in range(10):
        sample = generator[i]
        import ipdb

        ipdb.set_trace()
        rank_0_print(f"Sample {i}: {sample}")

    rank_0_print(f"Generated {preference_count} preference samples and {similarity_count} similarity samples")
    rank_0_print(
        f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.similarity_ratio:.1f} similarity"
    )


if __name__ == "__main__":
    test()
