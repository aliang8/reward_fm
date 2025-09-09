import random
from rfm.data.generators.pref import PreferenceDataGenerator
from rfm.data.generators.sim import SimilarityDataGenerator
from rfm.utils.logging import rank_0_print
from rfm.data.generators.base import BaseDataGenerator
from rfm.data.generators.vqa_progress import VQAProgressGenerator


class DataGenerator(BaseDataGenerator):
    """Data generator that combines preference and similarity generation."""

    def __init__(self, config, is_evaluation=False, **kwargs):
        """Initialize DataGenerator with configuration."""
        super().__init__(config, is_evaluation, **kwargs)

        # Initialize the individual generators
        self.preference_generator = PreferenceDataGenerator(config, is_evaluation, verbose=False)
        self.similarity_generator = SimilarityDataGenerator(config, is_evaluation, verbose=False)
        self.progress_generator = VQAProgressGenerator(config, is_evaluation, verbose=False)

        # Set the ratio for sampling between preference, similarity, and progress
        self.sample_type_ratio = config.sample_type_ratio

    def __next__(self):
        """Create a sample based on the configured ratios."""
        prob = random.random()
        if prob < self.sample_type_ratio[0]:
            return self.preference_generator.__next__()
        elif prob < self.sample_type_ratio[0] + self.sample_type_ratio[1]:
            return self.similarity_generator.__next__()
        else:
            return self.progress_generator.__next__()


def test():
    """Test the BatchCollator with generated samples."""
    # Create a mock config for testing
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class MockDataConfig:
        train_datasets: List[str] = None
        train_subsets: List[str] = None
        eval_datasets: List[str] = None
        eval_subsets: List[str] = None
        sample_type_ratio: List[float] = None
        shuffle: bool = True
        seed: int = 42
        num_proc: int = 4
        max_frames: int = 8  # Use 8 frames for testing the new subsampling logic
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: List[float] = None

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

    mock_config = MockConfig(data=mock_data_config, debug=False)

    # Create data generator with mock config
    generator = DataGenerator(config=mock_data_config)

    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")
    from rfm.data.dataset import InfiniteDataGeneratorDataset

    infinite_dataset = InfiniteDataGeneratorDataset(generator)

    preference_count = 0
    similarity_count = 0

    for i in range(10):
        sample = infinite_dataset[i]
        import ipdb

        ipdb.set_trace()
        if sample.sample_type == "preference":
            preference_count += 1
        else:
            similarity_count += 1
        rank_0_print(f"Sample {i}: {sample.sample_type}")

    rank_0_print(f"Generated {preference_count} preference samples and {similarity_count} similarity samples")
    rank_0_print(
        f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.similarity_ratio:.1f} similarity"
    )


if __name__ == "__main__":
    test()
