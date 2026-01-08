#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.
"""

import torch
from collections import Counter

from rfm.data.dataset_types import PreferenceSample, ProgressSample
from rfm.data.samplers.base import RFMBaseSampler
from rfm.utils.distributed import rank_0_print
from sentence_transformers import SentenceTransformer


class ConfusionMatrixSampler(RFMBaseSampler):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load sentence transformer model and precompute embeddings for all unique tasks
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        self.sentence_model.eval()

        # Precompute language embeddings for all unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Precomputing language embeddings for {len(unique_tasks)} unique tasks", verbose=self.verbose)
        self.task_embeddings = {}
        for task in unique_tasks:
            embedding = self.sentence_model.encode(task)
            self.task_embeddings[task] = torch.tensor(embedding)
        rank_0_print(f"Precomputed {len(self.task_embeddings)} language embeddings", verbose=self.verbose)

        # Free up the model after precomputation (no longer needed)
        del self.sentence_model

        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} confusion matrix sample indices from {len(self.robot_trajectories)} trajectories and {len(self.task_indices)} tasks"
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible task-trajectory pair sample indices."""
        sample_indices = []

        # Get unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Found {len(unique_tasks)} unique tasks: {unique_tasks}", verbose=self.verbose)

        trajectories_to_process = self.robot_trajectories
        rank_0_print(
            f"Processing {len(trajectories_to_process)} trajectories for confusion matrix analysis",
            verbose=self.verbose,
        )

        # Create all task-trajectory pairs
        for lang_task in unique_tasks:
            video_task_count = Counter()

            for traj_idx in trajectories_to_process:
                traj = self.dataset[traj_idx]
                video_task = traj["task"]

                # Limit the number of video trajectories for each task to 5
                if video_task_count[video_task] >= 5:
                    continue

                sample_indices.append({
                    "traj_idx": traj_idx,
                    "lang_task": lang_task,
                    "video_task": video_task,
                    "video_path": traj["frames"],
                    "id": traj["id"],
                })
                video_task_count[video_task] += 1

        rank_0_print(f"Generated {len(sample_indices)} task-trajectory pairs", verbose=self.verbose)
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> PreferenceSample:
        """Generate a single task-trajectory sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        lang_task = sample_idx_info["lang_task"]
        video_task = sample_idx_info["video_task"]
        video_path = sample_idx_info["video_path"]

        video_traj = self.dataset[traj_idx]

        # Look up precomputed embedding instead of encoding
        text_embedding = self.task_embeddings[lang_task]

        metadata = {
            "id": video_traj["id"],
            "lang_task": lang_task,
            "video_task": video_task,
            "video_path": video_path,
        }

        # Override task and text_embedding in the trajectory dict
        video_traj_with_task = video_traj.copy()
        video_traj_with_task["task"] = lang_task
        video_traj_with_task["text_embedding"] = text_embedding

        sample_trajectory = self._get_traj_from_data(
            traj=video_traj_with_task,
            metadata=metadata,
        )

        sample = ProgressSample(trajectory=sample_trajectory)
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
