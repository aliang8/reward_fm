import random
import torch

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.sim import SimSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.utils.distributed import rank_0_print


class RFMDataset(BaseDataset):
    """Dataset that combines preference, similarity, and progress generation."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        self.pref_sampler = PrefSampler(
            config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
        )
        self.progress_sampler = ProgressSampler(
            config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
        )
        self.similarity_sampler = SimSampler(
            config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
        )

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.data_len = len(self.dataset)

    def __len__(self):
        if self.max_samples is None:
            return max(self.data_len, self.batch_size)
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a data sample from the dataset."""
        idx = idx % self.data_len

        # Get the item from the dataset
        item = self.filtered_dataset[idx]

        # Preference-only override by data_source using raw filtered dataset entry
        pref_only = getattr(self.config, "pref_only_datasets", []) or []
        data_source = item.get("data_source", None)
        if data_source in pref_only:
            return self.pref_sampler._generate_sample(item)

        # Available samplers with their probabilities
        samplers = [
            ("pref", self.sample_type_ratio[0], self.pref_sampler),
            ("progress", self.sample_type_ratio[1], self.progress_sampler),
            ("similarity", self.sample_type_ratio[2], self.similarity_sampler),
        ]

        # Remove samplers with zero probability
        available_samplers = [(name, prob, sampler) for name, prob, sampler in samplers if prob > 0]

        # Fallback to progress sampler if no samplers have positive probability
        if not available_samplers:
            return self.progress_sampler._generate_sample(item)

        # Normalize probabilities
        total_prob = sum(prob for _, prob, _ in available_samplers)
        normalized_samplers = [(name, prob / total_prob, sampler) for name, prob, sampler in available_samplers]

        # Select sampler based on normalized probabilities
        prob = random.random()
        cumulative_prob = 0.0

        for name, normalized_prob, sampler in normalized_samplers:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return sampler._generate_sample(item)

        # Final fallback (should not reach here, but safety net)
        return self.progress_sampler._generate_sample(item)


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
    generator = RFMDataset(config=mock_data_config)

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
