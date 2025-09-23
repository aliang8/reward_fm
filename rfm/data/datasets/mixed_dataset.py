import random

from .base import RFMBaseDataset
from .pref import PrefDataset
from .sim import SimilarityDataset
from .vqa_progress import VQAProgressDataset
from rfm.utils.logging import rank_0_print


class MixedDataset(RFMBaseDataset):
    """Dataset that combines preference, similarity, and progress generation."""

    def __init__(self, config, is_evaluation=False, max_samples=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        # Initialize the individual datasets
        self.pref_dataset = PrefDataset(config, is_evaluation, verbose=False, **kwargs)
        self.similarity_dataset = SimilarityDataset(config, is_evaluation, verbose=False, **kwargs)
        self.progress_dataset = VQAProgressDataset(config, is_evaluation, verbose=False, **kwargs)

        # Set the ratio for sampling between preference, similarity, and progress
        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return max(len(self.pref_dataset), len(self.similarity_dataset), len(self.progress_dataset))
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a sample based on the configured ratios."""
        prob = random.random()
        if prob < self.sample_type_ratio[0]:
            return self.pref_dataset[idx]
        elif prob < self.sample_type_ratio[0] + self.sample_type_ratio[1]:
            return self.similarity_dataset[idx]
        else:
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
