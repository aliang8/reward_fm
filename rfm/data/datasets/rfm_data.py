import random

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
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = ProgressSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )
        if self.config.sample_type_ratio[2] > 0:
            self.similarity_sampler = SimSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.data_len = len(self.dataset)

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
        item = self.dataset[idx]
        return self._generate_sample_from_item(item)

    def _generate_sample_from_item(self, item):
        """Shared sampler logic that can be reused by balanced datasets."""
        data_source = item["data_source"]

        # Preference-only override by data_source using raw filtered dataset entry
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
        available_samplers = [
            (name, prob, sampler) for name, prob, sampler in samplers if prob > 0 and sampler is not None
        ]

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
                (name, prob, sampler) for name, prob, sampler in available_samplers if name not in tried_samplers
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