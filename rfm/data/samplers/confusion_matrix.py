#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.
"""

from tqdm import tqdm
import torch
from collections import Counter

from rfm.data.dataset_types import PreferenceSample, ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import DataGenStrat
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
)
from rfm.utils.distributed import rank_0_print
from sentence_transformers import SentenceTransformer


class ConfusionMatrixSampler(RFMBaseSampler):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.
    """

    def __init__(self, config, dataset, combined_indices, dataset_success_cutoff_map=None, is_evaluation=False, verbose=True, **kwargs):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0

        # if we are loading embeddings, we need to also load the sentence transformer model
        if self.config.load_embeddings:
            self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            self.sentence_model.eval()

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

        # Get the original trajectory
        video_traj = self.dataset[traj_idx]

        frames = None
        video_embeddings = None
        text_embedding = None
        max_frames = self.config.max_frames

        if self.config.load_embeddings and video_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(video_traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = self.sentence_model.encode(lang_task)
            text_embedding = torch.tensor(text_embedding)

            video_embeddings, _ = linspace_subsample_frames(video_embeddings, max_frames)
            frames_shape_orig = video_embeddings.shape
            video_embeddings, _ = pad_trajectory_to_max_frames_torch(video_embeddings, [0], max_frames)
        else:
            frames = self._get_trajectory_frames(traj_idx)
            if frames is None or len(frames) == 0:
                return None

            frames, _ = linspace_subsample_frames(frames, max_frames)
            frames_shape_orig = frames.shape
            frames, _ = pad_trajectory_to_max_frames_np(frames, [0], max_frames)

        # Create metadata for the confusion matrix analysis
        metadata = {
            "id": video_traj["id"],
            "lang_task": lang_task,
            "video_task": video_task,
            "video_path": video_path,
        }

        # Create trajectory for the sample (using the original trajectory data but with new task)
        sample_trajectory = Trajectory(
            id=video_traj["id"],
            task=lang_task,  # Use the confusion matrix task, not the original trajectory task
            frames=frames,
            frames_shape=frames_shape_orig,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            data_source=video_traj["data_source"],
            lang_vector=video_traj["lang_vector"],  # Keep original language vector
            is_robot=video_traj["is_robot"],
            quality_label=video_traj["quality_label"],
            data_gen_strategy=DataGenStrat.CONFUSION_MATRIX.value,
            target_progress=[
                1.0
            ],  # Assume trajectory is complete for confusion matrix, also don't really care about progress here
            metadata=metadata,
        )

        sample = ProgressSample(trajectory=sample_trajectory)

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
