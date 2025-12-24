#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.
"""

from typing import Dict, List, Any

from tqdm import tqdm
import torch
from collections import Counter

from rfm.data.dataset_types import PreferenceSample, ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import DataGenStrat
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    load_embeddings_from_path,
    load_frames_from_npz,
    create_trajectory_from_dict,
)
from rfm.utils.distributed import rank_0_print
from sentence_transformers import SentenceTransformer


class ConfusionMatrixSampler(RFMBaseSampler):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.
    """

    def __init__(
        self,
        **kwargs,
    ):
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

        frames = None
        video_embeddings = None
        text_embedding = None
        max_frames = self.config.max_frames

        if self.config.load_embeddings and video_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(video_traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            video_embeddings, frame_indices = linspace_subsample_frames(video_embeddings, max_frames)
            frames_shape_orig = video_embeddings.shape
        else:
            frames = load_frames_from_npz(video_traj["frames"])
            if frames is None or len(frames) == 0:
                return None

            frames, frame_indices = linspace_subsample_frames(frames, max_frames)
            frames_shape_orig = frames.shape

        # Create progress values for each subsampled frame (all 1.0 since trajectory is complete)
        num_subsampled = len(frame_indices)
        progress_values = [1.0] * num_subsampled

        # Look up precomputed embedding instead of encoding
        text_embedding = self.task_embeddings[lang_task]

        metadata = {
            "id": video_traj["id"],
            "lang_task": lang_task,
            "video_task": video_task,
            "video_path": video_path,
        }

        sample_trajectory = create_trajectory_from_dict(
            video_traj,
            overrides={
                "task": lang_task,  # Use the confusion matrix task, not the original trajectory task
                "frames": frames if not self.config.load_embeddings else None,
                "frames_shape": frames_shape_orig,
                "video_embeddings": video_embeddings if self.config.load_embeddings else None,
                "text_embedding": text_embedding,
                "data_gen_strategy": DataGenStrat.SUCCESSFUL.value,
                "target_progress": progress_values,
                "metadata": metadata,
            },
        )
        sample_trajectory = self._post_process_trajectory(sample_trajectory)

        sample = ProgressSample(trajectory=sample_trajectory)
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
