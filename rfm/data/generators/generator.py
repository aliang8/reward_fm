import random
from rfm.data.generators.pref import PreferenceDataGenerator
from rfm.data.generators.sim import SimilarityDataGenerator
from rfm.utils.logging import rank_0_print
from rfm.data.generators.base import BaseDataGenerator

class DataGenerator(BaseDataGenerator):
    """Data generator that combines preference and similarity generation."""

    def __init__(self, config, is_evaluation=False, **kwargs):
        """Initialize DataGenerator with configuration."""
        super().__init__(config, is_evaluation, **kwargs)
        
        # Initialize the individual generators
        self.preference_generator = PreferenceDataGenerator(config, is_evaluation, verbose=False)
        self.similarity_generator = SimilarityDataGenerator(config, is_evaluation, verbose=False)
        
        # Set the ratio for sampling between preference and similarity
        self.preference_ratio = config.preference_ratio
        self.similarity_ratio = 1.0 - config.preference_ratio
        
        rank_0_print(f"DataGenerator initialized with preference_ratio={self.preference_ratio:.2f}, similarity_ratio={self.similarity_ratio:.2f}")

    def _create_sample(self):
        """Create a sample based on the configured ratios."""
        if random.random() < self.preference_ratio:
            return self.preference_generator._create_preference_sample()
        else:
            return self.similarity_generator._create_similarity_sample()
    
    def _create_preference_sample(self):
        """Create a preference sample using the preference generator."""
        return self.preference_generator._create_preference_sample()
    
    def _create_similarity_sample(self):
        """Create a similarity sample using the similarity generator."""
        return self.similarity_generator._create_similarity_sample()

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
        preference_ratio: float = 1.0
        similarity_ratio: float = 0.0
        dataset_preference_ratio: float = 0.7
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
        train_datasets=["abraranwar/libero_rfm"],
        train_subsets=["libero256_90"],
        preference_ratio=1.0,
        similarity_ratio=0.0,
        preference_strategy_ratio=[0.8, 0.1, 0.1, 0.0],
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,  # Use 8 frames for testing the new subsampling logic
        force_reprocess=False,
        model_type="default"
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