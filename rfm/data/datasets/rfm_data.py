import collections
import random
import torch

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.sim import SimSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.data.dataset_category import is_preference_only
from rfm.utils.distributed import rank_0_print


class RFMDataset(BaseDataset):
    """Dataset that combines preference, similarity, and progress generation."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        self.pref_sampler = None
        self.progress_sampler = None
        self.similarity_sampler = None

        if self.config.sample_type_ratio[0] > 0:
            self.pref_sampler = PrefSampler(
                config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
            )
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = ProgressSampler(
                config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
            )
        if self.config.sample_type_ratio[2] > 0:
            self.similarity_sampler = SimSampler(
                config, self.dataset, self._combined_indices, self.dataset_success_cutoff_map, is_evaluation, verbose=False, **kwargs
            )

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.data_len = len(self.filtered_dataset)
        self._resample_attempt_stats: dict[str, collections.defaultdict[str, list[int]]] = {
            "preference": collections.defaultdict(list),
            "progress": collections.defaultdict(list),
            "similarity": collections.defaultdict(list),
        }
        self._resample_dataset_attempt_stats: dict[str, collections.defaultdict[str, list[int]]] = {
            "preference": collections.defaultdict(list),
            "progress": collections.defaultdict(list),
            "similarity": collections.defaultdict(list),
        }

    def _record_resample_attempt(
        self, sample_type: str, strategy: str, sample_attempts: int, dataset_attempts: int
    ) -> None:
        if sample_type not in self._resample_attempt_stats:
            return

        self._resample_attempt_stats[sample_type][strategy].append(sample_attempts)
        self._resample_dataset_attempt_stats[sample_type][strategy].append(dataset_attempts)

    def _set_resample_attempts(self, sample, dataset_attempts: int):
        if sample is None:
            return None
        dataset_attempts = max(1, int(dataset_attempts))

        sample_attempts = int(getattr(sample, "resample_attempts", dataset_attempts))
        sample_attempts = max(1, sample_attempts)
        sample.resample_attempts = sample_attempts

        sample_type = sample.sample_type
        strategy = str(sample.data_gen_strategy)
        self._record_resample_attempt(sample_type, strategy, sample_attempts, dataset_attempts)

        return sample

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats

    def __len__(self):
        if self.max_samples is None:
            return max(self.data_len, self.batch_size)
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a data sample from the dataset."""
        idx = idx % self.data_len

        # Get the item from the filtered dataset
        item = self.filtered_dataset[idx]

        # Preference-only override by data_source using raw filtered dataset entry
        data_source = item["data_source"]
        if is_preference_only(data_source) and self.pref_sampler is not None:
            sample = self.pref_sampler._generate_sample(item)
            if sample is not None:
                return self._set_resample_attempts(sample, 1)

        # Available samplers with their probabilities
        samplers = [
            ("pref", self.sample_type_ratio[0], self.pref_sampler),
            ("progress", self.sample_type_ratio[1], self.progress_sampler),
            ("similarity", self.sample_type_ratio[2], self.similarity_sampler),
        ]

        # Remove samplers with zero probability or None samplers
        available_samplers = [(name, prob, sampler) for name, prob, sampler in samplers if prob > 0 and sampler is not None]

        # Fallback to progress sampler if no samplers have positive probability
        if not available_samplers:
            if self.progress_sampler is not None:
                sample = self.progress_sampler._generate_sample(item)
                if sample is not None:
                    return self._set_resample_attempts(sample, 1)
            raise ValueError("No samplers available")

        # Try samplers until we get a non-None result
        max_attempts = len(available_samplers) * 2  # Try each sampler multiple times if needed
        attempt = 0
        tried_samplers = set()

        while attempt < max_attempts:
            attempt += 1

            # If we've tried all samplers, reset and try again
            if len(tried_samplers) >= len(available_samplers):
                tried_samplers.clear()

            # Filter out already tried samplers if we haven't exhausted all options
            remaining_samplers = [
                (name, prob, sampler) for name, prob, sampler in available_samplers
                if name not in tried_samplers
            ]

            # If no remaining samplers, reset and try all again
            if not remaining_samplers:
                tried_samplers.clear()
                remaining_samplers = available_samplers

            # Normalize probabilities for remaining samplers
            total_prob = sum(prob for _, prob, _ in remaining_samplers)
            if total_prob == 0:
                # Reset and try all samplers again
                tried_samplers.clear()
                remaining_samplers = available_samplers
                total_prob = sum(prob for _, prob, _ in remaining_samplers)

            normalized_samplers = [(name, prob / total_prob, sampler) for name, prob, sampler in remaining_samplers]

            # Select sampler based on normalized probabilities
            prob = random.random()
            cumulative_prob = 0.0
            selected_sampler = None
            selected_name = None

            for name, normalized_prob, sampler in normalized_samplers:
                cumulative_prob += normalized_prob
                if prob <= cumulative_prob:
                    selected_sampler = sampler
                    selected_name = name
                    break

            # Fallback: select first sampler if selection failed
            if selected_sampler is None:
                selected_name, _, selected_sampler = remaining_samplers[0]

            # Try the selected sampler
            sample = selected_sampler._generate_sample(item)

            # If sample is not None, return it
            if sample is not None:
                return self._set_resample_attempts(sample, attempt)
            
            # Sample is None, mark this sampler as tried
            tried_samplers.add(selected_name)

        # All attempts failed, try progress sampler as final fallback
        if self.progress_sampler is not None:
            sample = self.progress_sampler._generate_sample(item)
            if sample is not None:
                return self._set_resample_attempts(sample, attempt)

        # Final fallback: raise error if all samplers returned None
        raise ValueError(f"All samplers failed to generate a sample after {max_attempts} attempts")


def test():
    """Test the RFMDataset with generated samples and timing."""
    import time
    from collections import defaultdict
    from dataclasses import dataclass

    # Create a mock config for testing
    @dataclass
    class MockDataConfig:
        train_datasets: list[str] = None
        eval_datasets: list[str] = None
        dataset_type: str = "rfm"
        max_frames_after_preprocessing: int = 64
        max_frames: int = 8
        resized_height: int = 128
        resized_width: int = 128
        sample_type_ratio: list[float] = None
        dataset_preference_ratio: float = None
        preference_strategy_ratio: list[float] = None
        progress_strategy_ratio: list[float] = None
        similarity_strategy_ratio: list[float] = None
        data_source_weights: dict[str, float] = None
        shuffle: bool = True
        seed: int = 42
        eval_subset_size: int = None
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        num_bins: int = 10
        load_embeddings: bool = False
        progress_pred_type: str = "absolute"
        min_success: float = 0.8
        max_success: float = 0.95
        dataset_success_cutoff_file: str = None
        pairwise_progress: bool = False

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["jesbu1_oxe_rfm_oxe_jaco_play"],
        eval_datasets=["jesbu1_oxe_rfm_oxe_jaco_play"],
        dataset_type="rfm",
        max_frames=16,
        max_frames_after_preprocessing=64,
        resized_height=128,
        resized_width=128,
        sample_type_ratio=[0, 1, 0],  # pref, progress, similarity
        dataset_preference_ratio=0.7,
        preference_strategy_ratio=[0.8, 0.1, 0.1, 0.0],
        progress_strategy_ratio=[1, 0, 0, 0], # [successful, rewind, different_task, subsequence]
        similarity_strategy_ratio=[1, 1, 1],  # rewind, suboptimal_same_task, paired_human_robot
        data_source_weights=None,
        shuffle=True,
        seed=42,
        eval_subset_size=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=6,
        num_bins=10,
        load_embeddings=True,
        progress_pred_type="absolute",
        min_success=0.8,
        max_success=0.95,
        dataset_success_cutoff_file=None,
        pairwise_progress=False,
    )

    # Create dataset generator
    rank_0_print("Initializing RFMDataset...")
    init_start = time.time()
    generator = RFMDataset(config=mock_data_config, batch_size=64)
    #import ipdb; ipdb.set_trace()
    init_time = time.time() - init_start
    rank_0_print(f"Dataset initialization took {init_time:.2f} seconds")

    # Test the dataset
    rank_0_print("Testing RFMDataset...")

    sample_type_counts = {"pref": 0, "progress": 0, "similarity": 0}
    source_counts = defaultdict(int)

    # Quick sample type distribution check
    for i in range(100):
        sample = generator[i]
        # Determine sample type - fix to use correct attribute names
        if hasattr(sample, "chosen_trajectory"):
            sample_type_counts["pref"] += 1
            source_counts[sample.chosen_trajectory.data_source] += 1
        elif hasattr(sample, "ref_trajectory"):
            # SimilaritySample has ref_trajectory, sim_trajectory, diff_trajectory
            sample_type_counts["similarity"] += 1
            source_counts[sample.ref_trajectory.data_source] += 1
        elif hasattr(sample, "trajectory"):
            # ProgressSample has trajectory
            sample_type_counts["progress"] += 1
            source_counts[sample.trajectory.data_source] += 1

    rank_0_print(f"Sample type distribution: {sample_type_counts}")
    rank_0_print(f"Data source distribution: {dict(source_counts)}")
    rank_0_print(f"Total samples tested: {sum(sample_type_counts.values())}")

    # Test batch loading with DataLoader
    from torch.utils.data import DataLoader

    rank_0_print("\nTesting batch loading with DataLoader...")
    batch_size = 64
    num_batches_to_test = 1000

    # Create DataLoader
    dataloader = DataLoader(
        generator,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for accurate timing
        shuffle=False,
        collate_fn=lambda x: x,  # Simple identity collator for timing
    )

    # Measure actual batch loading by timing the iteration itself
    iter_times = []
    dataloader_iter = iter(dataloader)
    for i in range(num_batches_to_test):
        iter_start = time.time()
        batch = next(dataloader_iter)
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)

    print(iter_times)
    if iter_times:
        avg_batch_time = sum(iter_times) / len(iter_times)
        total_batch_time = sum(iter_times)
        rank_0_print(f"\nBatch Loading Timing Results (batch_size={batch_size}):")
        rank_0_print(f"  Average time per batch: {avg_batch_time * 1000:.2f} ms")
        rank_0_print(f"  Total time for {len(iter_times)} batches: {total_batch_time:.2f} seconds")
        rank_0_print(f"  Throughput: {batch_size / avg_batch_time:.2f} samples/second")
        rank_0_print(f"  Min batch time: {min(iter_times) * 1000:.2f} ms")
        rank_0_print(f"  Max batch time: {max(iter_times) * 1000:.2f} ms")


if __name__ == "__main__":
    test()
