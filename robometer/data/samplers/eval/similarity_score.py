#!/usr/bin/env python3
"""
Data generator for similarity score evaluation.

This generator creates similarity samples for evaluation:
- For each paired human-robot trajectory pair (same task), creates similarity samples
- For each pairing, samples N negative trajectories from other tasks
- Creates similarity samples: ref=human, traj_sim=robot (same task), traj_diff=negative (different task)
"""

from typing import Dict, List, Any

from robometer.data.dataset_types import SimilaritySample, Trajectory
from robometer.data.samplers.base import RFMBaseSampler
from robometer.utils.distributed import rank_0_print


class SimilarityScoreSampler(RBMBaseSampler):
    """
    Data generator that creates similarity samples for evaluation.

    For each paired human-robot trajectory pair (same task):
    - Creates similarity samples with ref=human, traj_sim=robot, traj_diff=negative (from different task)
    - Samples N negative trajectories from other tasks for each pairing
    """

    def __init__(self, num_negatives: int = 2, **kwargs):
        super().__init__(**kwargs)

        self.num_negatives = num_negatives
        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} similarity score sample indices from {len(self.paired_human_robot_by_task)} tasks",
            verbose=self.verbose,
        )

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate all possible similarity score sample indices."""
        sample_indices = []

        # Iterate through all tasks with paired human-robot data
        for task, paired_info in self.paired_human_robot_by_task.items():
            human_indices = paired_info["human"]
            robot_indices = paired_info["robot"]

            if not human_indices or not robot_indices:
                continue

            # Get all tasks except the current one for negative sampling
            other_tasks = [t for t in self.task_indices.keys() if t != task]

            if not other_tasks:
                continue

            # Limit number of human/robot trajectories considered per task to reduce combinatorics
            if len(human_indices) > 2:
                selected_humans = self._local_random.sample(human_indices, 2)
            else:
                selected_humans = human_indices

            if len(robot_indices) > 2:
                selected_robots = self._local_random.sample(robot_indices, 2)
            else:
                selected_robots = robot_indices

            # For each selected human-robot pair, create N samples (one per negative)
            for human_idx in selected_humans:
                for robot_idx in selected_robots:
                    # Sample N negative tasks (with replacement if needed)
                    negative_tasks = self._local_random.choices(other_tasks, k=self.num_negatives)

                    # Create one sample index entry per negative
                    for negative_task in negative_tasks:
                        negative_task_indices = self.task_indices.get(negative_task, [])
                        if not negative_task_indices:
                            continue

                        # Store the negative task, we'll sample a specific negative during generation
                        sample_indices.append({
                            "human_idx": human_idx,
                            "robot_idx": robot_idx,
                            "task": task,
                            "negative_task": negative_task,
                            "negative_task_indices": negative_task_indices,
                        })

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> SimilaritySample:
        """Generate a single similarity sample from stored indices."""
        human_idx = sample_idx_info["human_idx"]
        robot_idx = sample_idx_info["robot_idx"]
        task = sample_idx_info["task"]
        negative_task = sample_idx_info["negative_task"]
        negative_task_indices = sample_idx_info["negative_task_indices"]

        # Get human and robot trajectories
        human_traj = self.dataset[human_idx]
        robot_traj = self.dataset[robot_idx]

        # Sample a negative trajectory from the specified different task
        if not negative_task_indices:
            return None

        negative_idx = self._local_random.choice(negative_task_indices)
        negative_traj = self.dataset[negative_idx]

        # Create trajectories for the similarity sample
        ref_traj = self._create_trajectory_from_data(human_traj)
        sim_traj = self._create_trajectory_from_data(robot_traj)
        diff_traj = self._create_trajectory_from_data(negative_traj)

        # Create metadata
        metadata = {
            "task": task,
            "negative_task": negative_task,
            "human_id": human_traj["id"],
            "robot_id": robot_traj["id"],
            "negative_id": negative_traj["id"],
        }

        # Add metadata to trajectories
        if ref_traj.metadata is None:
            ref_traj.metadata = {}
        ref_traj.metadata.update(metadata)

        sample = SimilaritySample(
            ref_trajectory=ref_traj,
            sim_trajectory=sim_traj,
            diff_trajectory=diff_traj,
            data_gen_strategy="similarity_score_eval",
        )

        return sample

    def _create_trajectory_from_data(self, traj_data: dict) -> Trajectory:
        """Create a Trajectory object from dataset entry."""
        metadata = {
            "data_gen_strategy": "similarity_score_eval",
        }

        trajectory = self._get_traj_from_data(
            traj=traj_data,
            metadata=metadata,
        )

        return trajectory

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
