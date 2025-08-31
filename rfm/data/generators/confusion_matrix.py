#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.

This generator creates samples where each task is paired with each trajectory
to analyze how well the model can distinguish between different tasks and trajectories.
For each task-trajectory pair, it creates a sample with the trajectory frames
and the task language instruction.
"""

from rfm.data.batch_collator import PreferenceSample, Trajectory
from rfm.utils.logging import rank_0_print
from typing import Dict, List, Optional, Union
from rfm.data.generators.base import BaseDataGenerator
from tqdm import tqdm
import numpy as np
import random


class ConfusionMatrixGenerator(BaseDataGenerator):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.
    """

    def __init__(
        self, config, is_evaluation=False, verbose=True, max_trajectories: Optional[int] = None
    ):
        super().__init__(config, is_evaluation, verbose=verbose)

        self.max_trajectories = max_trajectories
        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0

        rank_0_print(
            f"Generated {len(self.sample_indices)} confusion matrix sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories and {len(self.task_indices)} tasks"
        )

    def _generate_all_sample_indices(self) -> List[Dict]:
        """Generate all possible task-trajectory pair sample indices."""
        sample_indices = []

        # Get unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Found {len(unique_tasks)} unique tasks: {unique_tasks}")

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None:
            trajectories_to_process = self.robot_trajectories[:self.max_trajectories]

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
                    "trajectory_task": traj.get("task", "unknown")  # Original task of trajectory
                })

        rank_0_print(f"Generated {len(sample_indices)} task-trajectory pairs")
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single task-trajectory sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        task = sample_idx_info["task"]
        trajectory_task = sample_idx_info["trajectory_task"]

        # Get the original trajectory
        original_traj = self.dataset[traj_idx]

        # Get frames and create sample
        frames = self._get_trajectory_frames(traj_idx)
        if frames is None or len(frames) == 0:
            return None

        # Get max_frames from config
        max_frames = self.config.max_frames

        # Uniform subsample to max_frames
        frames, _ = self._linspace_subsample_frames(frames, max_frames)

        # Use the existing helper function to pad/subsample frames
        padded_frames, _ = self._pad_trajectory_to_max_frames(frames, [0], max_frames)

        # Create metadata for the confusion matrix analysis
        metadata = {
            "confusion_matrix_task": task,  # The task we're testing with
            "trajectory_original_task": trajectory_task,  # Original task of the trajectory
            "is_matching_task": task == trajectory_task,  # Whether task matches trajectory
            "data_gen_strategy": "confusion_matrix",
            "num_frames": len(frames),
            "max_frames": max_frames,
        }

        # Create trajectory for the sample (using the original trajectory data but with new task)
        sample_trajectory = Trajectory(
            id=original_traj["id"],
            task=task,  # Use the confusion matrix task, not the original trajectory task
            frames=padded_frames,
            frames_shape=padded_frames.shape,
            data_source=original_traj["data_source"],
            lang_vector=original_traj["lang_vector"],  # Keep original language vector
            is_robot=original_traj["is_robot"],
            quality_label=original_traj["quality_label"],
            data_gen_strategy="confusion_matrix",
            target_progress=[1.0],  # Assume trajectory is complete for confusion matrix
            metadata=metadata,
        )

        # Create a dummy "rejected" trajectory (same as chosen for this analysis)
        # We only care about the task discrimination, not preference
        rejected_trajectory = Trajectory(
            id=original_traj["id"],
            task=task,
            frames=padded_frames,
            frames_shape=padded_frames.shape,
            data_source=original_traj["data_source"],
            lang_vector=original_traj["lang_vector"],
            is_robot=original_traj["is_robot"],
            quality_label=original_traj["quality_label"],
            data_gen_strategy="confusion_matrix",
            target_progress=[1.0],
            metadata=metadata,
        )

        # Create preference sample (chosen and rejected are the same for confusion matrix)
        sample = PreferenceSample(
            chosen_trajectory=sample_trajectory,
            rejected_trajectory=rejected_trajectory,
            sample_type="preference"
        )

        return sample

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        """Get the next sample by generating it from stored indices."""
        if self.current_idx >= len(self.sample_indices):
            raise StopIteration

        # Get the sample indices for this sample
        sample_idx_info = self.sample_indices[self.current_idx]

        # Generate the actual sample on-demand
        sample = self._generate_sample_from_indices(sample_idx_info)

        # Skip invalid samples
        while sample is None and self.current_idx < len(self.sample_indices):
            self.current_idx += 1
            if self.current_idx >= len(self.sample_indices):
                raise StopIteration

            sample_idx_info = self.sample_indices[self.current_idx]
            sample = self._generate_sample_from_indices(sample_idx_info)

        if sample is None:
            raise StopIteration

        self.current_idx += 1
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.__next__()