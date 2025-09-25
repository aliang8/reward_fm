#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.
"""

from tqdm import tqdm

from rfm.data.dataset_types import PreferenceSample, ProgressSample, Trajectory
from .base import RFMBaseDataset
from .pref import DataGenStrat
from .helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
)
from rfm.utils.distributed import rank_0_print


class ConfusionMatrixDataset(RFMBaseDataset):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.
    """

    def __init__(self, config, is_evaluation=False, verbose=True, max_trajectories: int | None = None):
        super().__init__(config, is_evaluation, verbose=verbose)

        self.max_trajectories = max_trajectories
        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0

        rank_0_print(
            f"Generated {len(self.sample_indices)} confusion matrix sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories and {len(self.task_indices)} tasks"
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible task-trajectory pair sample indices."""
        sample_indices = []

        # Get unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Found {len(unique_tasks)} unique tasks: {unique_tasks}")

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None:
            trajectories_to_process = self.robot_trajectories[: self.max_trajectories]

        rank_0_print(f"Processing {len(trajectories_to_process)} trajectories for confusion matrix analysis")

        # Create all task-trajectory pairs
        for task in tqdm(unique_tasks, desc="Generating task-trajectory pairs"):
            for traj_idx in trajectories_to_process:
                traj = self.dataset[traj_idx]

                # Get trajectory length from frames
                frames_path = traj.get("frames")
                if not frames_path:
                    continue

                # Store the pairing information
                sample_indices.append({
                    "traj_idx": traj_idx,
                    "task": task,
                    "trajectory_task": traj.get("task", "unknown"),  # Original task of trajectory
                })

        rank_0_print(f"Generated {len(sample_indices)} task-trajectory pairs")
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> PreferenceSample:
        """Generate a single task-trajectory sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        task = sample_idx_info["task"]
        trajectory_task = sample_idx_info["trajectory_task"]

        # Get the original trajectory
        original_traj = self.dataset[traj_idx]

        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None

        # Get max_frames from config
        max_frames = self.config.max_frames

        if self.config.load_embeddings and original_traj.get("embeddings_path"):
            # Load embeddings from path
            video_embeddings = load_embeddings_from_path(original_traj["embeddings_path"], "video_embeddings")
            text_embedding = load_embeddings_from_path(original_traj["embeddings_path"], "text_embedding")

            # Uniform subsample to max_frames
            video_embeddings, _ = linspace_subsample_frames(video_embeddings, max_frames)

            # Use the torch padding function for embeddings
            video_embeddings, _ = pad_trajectory_to_max_frames_torch(video_embeddings, [0], max_frames)
        else:
            # Get frames and create sample
            frames = self._get_trajectory_frames(traj_idx)
            if frames is None or len(frames) == 0:
                return None

            # Uniform subsample to max_frames
            frames, _ = linspace_subsample_frames(frames, max_frames)

            # Use the existing helper function to pad/subsample frames
            frames, _ = pad_trajectory_to_max_frames_np(frames, [0], max_frames)

        # Create metadata for the confusion matrix analysis
        metadata = {
            "confusion_matrix_task": task,
            "trajectory_original_task": trajectory_task,  # Original task of the trajectory
        }

        # Create trajectory for the sample (using the original trajectory data but with new task)
        sample_trajectory = Trajectory(
            id=original_traj["id"],
            task=task,  # Use the confusion matrix task, not the original trajectory task
            frames=frames,
            frames_shape=frames.shape if frames is not None and hasattr(frames, "shape") else None,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            data_source=original_traj["data_source"],
            lang_vector=original_traj["lang_vector"],  # Keep original language vector
            is_robot=original_traj["is_robot"],
            quality_label=original_traj["quality_label"],
            data_gen_strategy=DataGenStrat.CONFUSION_MATRIX.value,
            target_progress=[1.0],  # Assume trajectory is complete for confusion matrix
            metadata=metadata,
        )

        sample = ProgressSample(trajectory=sample_trajectory)

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
