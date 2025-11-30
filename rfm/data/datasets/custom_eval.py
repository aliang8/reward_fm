#!/usr/bin/env python3
import torch

from rfm.data.datasets.base import BaseDataset
from rfm.data.samplers import *


class CustomEvalDataset(BaseDataset):
    """Dataset that wraps a sampler for custom evaluation purposes."""

    def __init__(self, sampler_type, config, is_evaluation=False, verbose=True, **kwargs):
        """Initialize custom eval dataset with a sampler type.

        Args:
            sampler_type: Type of sampler to create (e.g., "confusion_matrix", "reward_alignment", "policy_ranking", "success_failure")
            config: Configuration object
            is_evaluation: Whether this is for evaluation
            verbose: Verbose flag
            **kwargs: Additional keyword arguments for the sampler
        """
        super().__init__(config, is_evaluation, verbose=verbose, **kwargs)

        sampler_cls = {
            "confusion_matrix": ConfusionMatrixSampler,
            "reward_alignment": RewardAlignmentSampler,
            "policy_ranking": ProgressDefaultSampler,
            "quality_preference": QualityPreferenceSampler,
            "similarity_score": SimilarityScoreSampler,
        }

        if "roboarena" in self.config.data.eval_datasets:
            sampler_cls["quality_preference"] = RoboArenaQualityPreferenceSampler

        if sampler_type not in sampler_cls:
            raise ValueError(f"Unknown sampler type: {sampler_type}. Available: {list(sampler_cls.keys())}")

        self.sampler = sampler_cls[sampler_type](
            config,
            self.dataset,
            self._combined_indices,
            self.dataset_success_cutoff_map,
            is_evaluation=is_evaluation,
            verbose=verbose,
            **kwargs,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]
