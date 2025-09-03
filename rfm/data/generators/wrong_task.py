#!/usr/bin/env python3
"""
Data generator for wrong task accuracy analysis.
"""

from rfm.data.dataset_types import PreferenceSample, Trajectory
from rfm.utils.logging import rank_0_print
from typing import Dict, List, Optional, Union
from rfm.data.generators.base import BaseDataGenerator
from tqdm import tqdm
import numpy as np
import random


class WrongTaskGenerator(BaseDataGenerator):
    """
    Data generator that creates samples for wrong task accuracy analysis.

    For each trajectory, creates preference samples by pairing it with N different
    trajectories from different tasks to measure preference accuracy when the
    model should prefer the original trajectory over wrong task trajectories.
    """

    def __init__(self, config, is_evaluation=False, verbose=True, max_trajectories: Optional[int] = None):
        super().__init__(config, is_evaluation, verbose=verbose)

        self.max_trajectories = max_trajectories
        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0

        rank_0_print(
            f"Generated {len(self.sample_indices)} wrong task preference sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories"
        )

    def _generate_all_sample_indices(self) -> List[Dict]:
        """Generate all possible wrong task preference sample indices."""
        sample_indices = []

        # Get unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Found {len(unique_tasks)} unique tasks: {unique_tasks}")

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None:
            trajectories_to_process = self.robot_trajectories[: self.max_trajectories]

        rank_0_print(f"Processing {len(trajectories_to_process)} trajectories for wrong task preference analysis")

        # For each trajectory, create preference samples with wrong task trajectories
        for traj_idx in tqdm(trajectories_to_process, desc="Generating wrong task preference pairs"):
            traj = self.dataset[traj_idx]
            original_task = traj.get("task", "unknown")

            # Get frames path to check if trajectory is valid
            frames_path = traj.get("frames")
            if not frames_path:
                continue

            # Find trajectories from different tasks
            wrong_task_trajectories = []
            for other_traj_idx in self.robot_trajectories:
                if other_traj_idx == traj_idx:  # Skip same trajectory
                    continue

                other_traj = self.dataset[other_traj_idx]
                other_task = other_traj.get("task", "unknown")

                # Only include trajectories from different tasks
                if other_task != original_task:
                    wrong_task_trajectories.append(other_traj_idx)

            # Sample N wrong task trajectories
            if len(wrong_task_trajectories) >= self.config.n_wrong_tasks:
                sampled_wrong_trajectories = random.sample(wrong_task_trajectories, self.config.n_wrong_tasks)
            else:
                # If not enough wrong task trajectories, use all available
                sampled_wrong_trajectories = wrong_task_trajectories

            # Create preference samples
            for wrong_traj_idx in sampled_wrong_trajectories:
                sample_indices.append(
                    {
                        "chosen_traj_idx": traj_idx,  # Original trajectory (should be preferred)
                        "rejected_traj_idx": wrong_traj_idx,  # Wrong task trajectory (should be rejected)
                        "original_task": original_task,
                        "wrong_task": self.dataset[wrong_traj_idx].get("task", "unknown"),
                    }
                )

        rank_0_print(f"Generated {len(sample_indices)} wrong task preference pairs")
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single wrong task preference sample from stored indices."""
        chosen_traj_idx = sample_idx_info["chosen_traj_idx"]
        rejected_traj_idx = sample_idx_info["rejected_traj_idx"]
        original_task = sample_idx_info["original_task"]
        wrong_task = sample_idx_info["wrong_task"]

        # Get the original trajectories
        chosen_traj = self.dataset[chosen_traj_idx]
        rejected_traj = self.dataset[rejected_traj_idx]

        # Get frames for both trajectories
        chosen_frames = self._get_trajectory_frames(chosen_traj_idx)
        rejected_frames = self._get_trajectory_frames(rejected_traj_idx)

        if chosen_frames is None or len(chosen_frames) == 0:
            return None
        if rejected_frames is None or len(rejected_frames) == 0:
            return None

        # Get max_frames from config
        max_frames = self.config.max_frames

        # Uniform subsample to max_frames for both trajectories
        chosen_frames, _ = self._linspace_subsample_frames(chosen_frames, max_frames)
        rejected_frames, _ = self._linspace_subsample_frames(rejected_frames, max_frames)

        # Use the existing helper function to pad/subsample frames
        chosen_padded_frames, _ = self._pad_trajectory_to_max_frames(chosen_frames, [0], max_frames)
        rejected_padded_frames, _ = self._pad_trajectory_to_max_frames(rejected_frames, [0], max_frames)

        # Create metadata for the wrong task preference analysis
        metadata = {
            "original_task": original_task,
            "wrong_task": wrong_task,
            "chosen_trajectory_id": chosen_traj["id"],
            "rejected_trajectory_id": rejected_traj["id"],
        }

        # Create trajectories for the preference sample
        chosen_trajectory = Trajectory(
            id=chosen_traj["id"],
            task=original_task,  # Keep original task
            frames=chosen_padded_frames,
            frames_shape=chosen_padded_frames.shape,
            data_source=chosen_traj["data_source"],
            lang_vector=chosen_traj["lang_vector"],
            is_robot=chosen_traj["is_robot"],
            quality_label=chosen_traj["quality_label"],
            data_gen_strategy="wrong_task_preference",
            target_progress=[1.0],  # Assume trajectory is complete
            metadata=metadata,
        )

        rejected_trajectory = Trajectory(
            id=rejected_traj["id"],
            task=wrong_task,  # Wrong task
            frames=rejected_padded_frames,
            frames_shape=rejected_padded_frames.shape,
            data_source=rejected_traj["data_source"],
            lang_vector=rejected_traj["lang_vector"],
            is_robot=rejected_traj["is_robot"],
            quality_label=rejected_traj["quality_label"],
            data_gen_strategy="wrong_task_preference",
            target_progress=[1.0],  # Assume trajectory is complete
            metadata=metadata,
        )

        # Create preference sample (chosen should be preferred over rejected)
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            metadata=metadata,
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
