import random
from collections import defaultdict

from rfm.data.datasets.mixed_dataset import MixedDataset
from rfm.utils.distributed import rank_0_print


class BalancedMixedDataset(MixedDataset):
    """Dataset that extends MixedDataset with configurable sampling weights per data source."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, max_samples, batch_size, **kwargs)

        self.data_source_weights = config.data_source_weights
        self.data_len = 10000000

        rank_0_print("Building source indices for BalancedMixedDataset...")
        self.source_indices = defaultdict(list)

        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
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

        # Fallback (should not reach here)
        if available_datasets:
            return self._get_balanced_sample(available_datasets[0][0])
        else:
            raise ValueError("No available datasets to sample from")

    def _get_balanced_sample(self, sample_type: str):
        """Get a sample of the specified type with weighted data source sampling."""
        # Select data source based on unified normalized weights
        selected_source = self._select_weighted_source()

        # Get available trajectory indices for this source
        source_indices = self.source_indices[selected_source]

        # Select trajectory index randomly within the source
        selected_traj_idx = random.choice(source_indices)

        # Enforce preference-only sampling for configured data sources
        pref_only = getattr(self.config, "pref_only_datasets", []) or []
        if selected_source in pref_only:
            return self.pref_dataset[selected_traj_idx]

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

        # Fallback (should not reach here)
        return available_sources[-1]


def test():
    """Test the BalancedMixedDataset with generated samples and timing."""
    import time
    from dataclasses import dataclass

    # Create a mock config for testing
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
        max_frames_after_preprocessing: int = 64
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: list[float] = None
        progress_strategy_ratio: list[float] = None
        similarity_strategy_ratio: list[float] = None
        dataset_preference_ratio: float = None
        data_source_weights: dict[str, float] = None
        load_embeddings: bool = False
        progress_pred_type: str = "absolute"
        max_success: float = 0.9
        pairwise_progress: bool = False
        pref_only_datasets: list[str] = None
        batch_size: int = 64

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        # train_datasets=["jesbu1_roboarena_0825_rfm_roboarena", "jesbu1_fino_net_rfm_fino_net", "jesbu1_failsafe_rfm_failsafe", "jesbu1_soar_rfm_soar_rfm"],
        # train_datasets=["jesbu1_failsafe_rfm_failsafe"],
        # train_datasets=["jesbu1_h2r_rfm_h2r", "anqil_rh20t_subset_rfm_rh20t_human", "anqil_rh20t_subset_rfm_rh20t_robot", "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm", "jesbu1_motif_rfm_motif_rfm"],
        # train_datasets=["jesbu1_motif_rfm_motif_rfm", "anqil_rh20t_subset_rfm_rh20t_human", "anqil_rh20t_subset_rfm_rh20t_robot", "jesbu1_h2r_rfm_h2r"],
        train_datasets=["jesbu1_motif_rfm_motif_rfm", "jesbu1_h2r_rfm_h2r"],

        sample_type_ratio=[0, 0, 1],  # pref, progress, similarity
        preference_strategy_ratio=[6, 1, 1, 0],
        progress_strategy_ratio=[1, 6, 1],
        similarity_strategy_ratio=[1, 1, 1],  # rewind, suboptimal_same_task, paired_human_robot
        dataset_preference_ratio=0.7,
        data_source_weights={
            "roboarena": 1,
            "fino_net": 1,
            "failsafe": 1,
        },
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,
        max_frames_after_preprocessing=64,
        force_reprocess=False,
        model_type="default",
        load_embeddings=True,
        progress_pred_type="absolute",
        max_success=0.9,
        pairwise_progress=False,
        pref_only_datasets=[],
        batch_size=64,
    )

    # Create balanced dataset generator
    rank_0_print("Initializing BalancedMixedDataset...")
    init_start = time.time()
    generator = BalancedMixedDataset(config=mock_data_config, batch_size=mock_data_config.batch_size)
    init_time = time.time() - init_start
    rank_0_print(f"Dataset initialization took {init_time:.2f} seconds")

    # Test the dataset
    rank_0_print("Testing BalancedMixedDataset...")

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
    num_batches_to_test = 100

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
