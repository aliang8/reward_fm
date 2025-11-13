import random
from collections import defaultdict

from rfm.data.datasets.rfm_data import RFMDataset
from rfm.data.dataset_category import is_preference_only
from rfm.utils.distributed import rank_0_print


class BalancedRFMDataset(RFMDataset):
    """Dataset that extends RFMDataset with configurable sampling weights per data source."""

    def __init__(self, config, is_evaluation=False, max_samples=None, batch_size=None, **kwargs):
        super().__init__(config, is_evaluation, max_samples, batch_size, **kwargs)

        self.data_source_weights = config.data_source_weights
        self.data_len = 10000000

        rank_0_print("Building source indices for BalancedRFMDataset...")
        self.source_indices = defaultdict(list)

        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        self._normalize_data_source_weights()

        if self.verbose:
            rank_0_print(f"BalancedRFMDataset initialized with {len(self.source_indices)} data sources")
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
        # Select data source based on unified normalized weights
        selected_source = self._select_weighted_source()

        # Get available trajectory indices for this source
        source_indices = self.source_indices[selected_source]

        # Select trajectory index randomly within the source
        selected_traj_idx = random.choice(source_indices)

        # Get the trajectory item from the dataset
        item = self.dataset[selected_traj_idx]

        # Preference-only override by data_source using raw dataset entry
        if is_preference_only(selected_source) and self.pref_sampler is not None:
            try:
                sample = self.pref_sampler._generate_sample(item)
                # If pref sampler returns None, fall through to try other samplers
                if sample is not None:
                    return sample
            except (ValueError, RuntimeError) as e:
                # If sampler raises an error, treat as None and try other samplers
                pass

        # Available sampler types with their probabilities
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
                try:
                    return self.progress_sampler._generate_sample(item)
                except (ValueError, RuntimeError) as e:
                    pass
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

            # Try the selected sampler with balanced data source sampling
            try:
                sample = self._get_balanced_sample_with_item(selected_name, item)
            except (ValueError, RuntimeError) as e:
                # If sampler raises an error, treat as None and try other samplers
                sample = None
            
            # If sample is not None, return it
            if sample is not None:
                return sample
            
            # Sample is None, mark this sampler as tried
            tried_samplers.add(selected_name)

        # All attempts failed, try progress sampler as final fallback
        if self.progress_sampler is not None:
            try:
                sample = self._get_balanced_sample_with_item("progress", item)
                if sample is not None:
                    return sample
            except (ValueError, RuntimeError) as e:
                pass

        # Final fallback: raise error if all samplers returned None
        raise ValueError(f"All samplers failed to generate a sample after {max_attempts} attempts")

    def _get_balanced_sample_with_item(self, sample_type: str, item):
        """Get a sample of the specified type using a pre-selected item."""
        # Generate sample using the appropriate sampler
        if sample_type == "pref":
            return self.pref_sampler._generate_sample(item)
        elif sample_type == "progress":
            return self.progress_sampler._generate_sample(item)
        elif sample_type == "similarity":
            return self.similarity_sampler._generate_sample(item)

    def _get_balanced_sample(self, sample_type: str):
        """Get a sample of the specified type with weighted data source sampling."""
        # Select data source based on unified normalized weights
        selected_source = self._select_weighted_source()

        # Get available trajectory indices for this source
        source_indices = self.source_indices[selected_source]

        # Select trajectory index randomly within the source
        selected_traj_idx = random.choice(source_indices)

        # Get the trajectory item from the dataset
        item = self.dataset[selected_traj_idx]

        # Enforce preference-only sampling for configured data sources
        if is_preference_only(selected_source):
            return self.pref_sampler._generate_sample(item)

        # Generate sample using the appropriate sampler
        return self._get_balanced_sample_with_item(sample_type, item)

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
    """Test the BalancedRFMDataset with generated samples and timing."""
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
        max_frames: int = 16
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
        # train_datasets=["jesbu1_motif_rfm_motif_rfm", "jesbu1_h2r_rfm_h2r"],
        # train_datasets=["anqil_rh20t_subset_rfm_rh20t_human", "anqil_rh20t_subset_rfm_rh20t_robot"],
        train_datasets=["jesbu1_oxe_rfm_oxe_jaco_play"],
        sample_type_ratio=[0, 1, 0],  # pref, progress, similarity
        preference_strategy_ratio=[6, 1, 1, 0],
        progress_strategy_ratio=[1, 0, 0], # default success, rewind, different task
        similarity_strategy_ratio=[1, 1, 1],  # rewind, suboptimal_same_task, paired_human_robot
        dataset_preference_ratio=0.7,
        data_source_weights={
            "roboarena": 1,
            "fino_net": 1,
            "failsafe": 1,
        },
        shuffle=True,
        seed=42,
        num_proc=0,
        max_frames=16,
        max_frames_after_preprocessing=64,
        force_reprocess=False,
        model_type="default",
        load_embeddings=True,
        progress_pred_type="absolute",
        max_success=0.9,
        pairwise_progress=False,
        batch_size=64,
    )

    # Create balanced dataset generator
    rank_0_print("Initializing BalancedRFMDataset...")
    init_start = time.time()
    generator = BalancedRFMDataset(config=mock_data_config, batch_size=mock_data_config.batch_size)
    init_time = time.time() - init_start
    rank_0_print(f"Dataset initialization took {init_time:.2f} seconds")

    # Test the dataset
    rank_0_print("Testing BalancedRFMDataset...")

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
