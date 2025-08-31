#!/usr/bin/env python3
"""
SimilarityDataGenerator class for producing batches of similarity scoring data.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from rfm.data.dataset_types import SimilaritySample, Trajectory
from rfm.data.generators.base import BaseDataGenerator
from rfm.utils.logging import rank_0_print


class SimilarityDataGenerator(BaseDataGenerator):
    """Data generator for producing batches of similarity scoring data."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        """Initialize SimilarityDataGenerator with configuration."""
        # Call parent constructor to load datasets
        super().__init__(config, is_evaluation, verbose=verbose)

        rank_0_print(f"SimilarityDataGenerator initialized with {len(self.dataset)} total trajectories")

    def _create_similarity_sample(self) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.

        Two modes (50/50 split):
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
        """

        # Randomly choose between rewind mode and optimal/suboptimal mode
        use_rewind_mode = random.choice([True, False])

        if use_rewind_mode:
            return self._create_rewind_similarity_sample()
        else:
            return self._create_optimal_similarity_sample()

    def _create_rewind_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using rewind logic.

        Rules:
        - traj_sim is rewound trajectory from same task as o^ref (different trajectory)
        - traj_diff MUST be from different task than o^ref
        - o^ref is optimal trajectory from a random task
        """

        # Get a random task and optimal trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))
        optimal_idx = random.choice(self.optimal_by_task[task_name])
        ref_traj = self.dataset[optimal_idx]

        # Create traj_sim: rewound trajectory from same task as o^ref
        traj_sim = self._create_rewind_trajectory(ref_traj)

        # Create traj_diff: trajectory from different task than o^ref
        other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]
        if not other_tasks:
            # If only one task available, use suboptimal from same task
            same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
            if same_task_suboptimal_indices:
                traj_diff_idx = random.choice(same_task_suboptimal_indices)
                traj_diff = self.dataset[traj_diff_idx]
            else:
                # Fallback: create another rewind trajectory
                traj_diff = self._create_rewind_trajectory(ref_traj)
        else:
            # Use trajectory from different task
            other_task = random.choice(other_tasks)
            other_task_indices = self.optimal_by_task[other_task]
            if other_task_indices:
                other_idx = random.choice(other_task_indices)
                traj_diff = self.dataset[other_idx]
            else:
                # Fallback: use suboptimal from same task
                same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
                if same_task_suboptimal_indices:
                    traj_diff_idx = random.choice(same_task_suboptimal_indices)
                    traj_diff = self.dataset[traj_diff_idx]
                else:
                    # Final fallback: create another rewind trajectory
                    traj_diff = self._create_rewind_trajectory(ref_traj)

        # Load frames and subsample if needed
        if isinstance(ref_traj["frames"], str):
            ref_frames = self._load_frames_from_npz(ref_traj["frames"])
        else:
            ref_frames = ref_traj["frames"]

        if isinstance(traj_sim["frames"], str):
            traj_sim_frames = self._load_frames_from_npz(traj_sim["frames"])
        else:
            traj_sim_frames = traj_sim["frames"]

        if isinstance(traj_diff["frames"], str):
            traj_diff_frames = self._load_frames_from_npz(traj_diff["frames"])
        else:
            traj_diff_frames = traj_diff["frames"]

        # Apply uniform subsampling to get consistent frame counts
        num_frames_to_sample = getattr(self.config, "max_frames", 8)

        # For reference trajectory, use uniform subsampling
        ref_frames, ref_indices = self._linspace_subsample_frames(ref_frames, num_frames_to_sample)
        ref_progress = [idx / (len(ref_frames) - 1) for idx in ref_indices]

        # For traj_sim (rewound), it's already subsampled
        if traj_sim.get("target_progress"):
            traj_sim_progress = traj_sim["target_progress"]
        else:
            traj_sim_progress = [i / (len(traj_sim_frames) - 1) for i in range(len(traj_sim_frames))]

        # For traj_diff, use uniform subsampling
        traj_diff_frames, traj_diff_indices = self._linspace_subsample_frames(traj_diff_frames, num_frames_to_sample)
        traj_diff_progress = [idx / (len(traj_diff_frames) - 1) for idx in traj_diff_indices]

        # Pad all trajectories to max_frames if needed
        ref_frames, ref_progress = self._pad_trajectory_to_max_frames(ref_frames, ref_progress, num_frames_to_sample)
        traj_sim_frames, traj_sim_progress = self._pad_trajectory_to_max_frames(
            traj_sim_frames, traj_sim_progress, num_frames_to_sample
        )
        traj_diff_frames, traj_diff_progress = self._pad_trajectory_to_max_frames(
            traj_diff_frames, traj_diff_progress, num_frames_to_sample
        )

        # Create SimilaritySample
        sample = SimilaritySample(
            reference_trajectory=Trajectory(
                frames=ref_frames,
                frames_shape=ref_frames.shape,
                id=ref_traj["id"],
                task=ref_traj["task"],
                lang_vector=ref_traj["lang_vector"],
                data_source=ref_traj["data_source"],
                quality_label=ref_traj.get("quality_label"),
                is_robot=ref_traj["is_robot"],
                target_progress=ref_progress,
                metadata={"data_gen_strategy": "rewind_similarity"},
            ),
            traj_sim_trajectory=Trajectory(
                frames=traj_sim_frames,
                frames_shape=traj_sim_frames.shape,
                id=traj_sim["id"],
                task=traj_sim["task"],
                lang_vector=traj_sim["lang_vector"],
                data_source=traj_sim["data_source"],
                quality_label=traj_sim["quality_label"],
                is_robot=traj_sim["is_robot"],
                target_progress=traj_sim_progress,
                metadata=traj_sim.get("metadata", {}),
            ),
            traj_diff_trajectory=Trajectory(
                frames=traj_diff_frames,
                frames_shape=traj_diff_frames.shape,
                id=traj_diff["id"],
                task=traj_diff["task"],
                lang_vector=traj_diff["lang_vector"],
                data_source=traj_diff["data_source"],
                quality_label=traj_diff["quality_label"],
                is_robot=traj_diff["is_robot"],
                target_progress=traj_diff_progress,
                metadata={"data_gen_strategy": "rewind_similarity"},
            ),
            data_gen_strategy="rewind_similarity",
        )

        return sample

    def _create_optimal_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using optimal/suboptimal logic.

        Rules:
        - o^ref is optimal trajectory from a random task
        - traj_sim is optimal trajectory from same task as o^ref (different trajectory)
        - traj_diff is suboptimal trajectory from same task as o^ref
        """

        # Get a random task and optimal trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))
        optimal_idx = random.choice(self.optimal_by_task[task_name])
        ref_traj = self.dataset[optimal_idx]

        # Create traj_sim: optimal trajectory from same task as o^ref (different trajectory)
        same_task_optimal_indices = [idx for idx in self.optimal_by_task[task_name] if idx != optimal_idx]
        if same_task_optimal_indices:
            traj_sim_idx = random.choice(same_task_optimal_indices)
            traj_sim = self.dataset[traj_sim_idx]
        else:
            # If no other optimal trajectories, use the same one
            traj_sim = ref_traj

        # Create traj_diff: suboptimal trajectory from same task as o^ref
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if same_task_suboptimal_indices:
            traj_diff_idx = random.choice(same_task_suboptimal_indices)
            traj_diff = self.dataset[traj_diff_idx]
        else:
            # If no suboptimal trajectories, create a rewind trajectory
            traj_diff = self._create_rewind_trajectory(ref_traj)

        # Load frames and subsample if needed
        if isinstance(ref_traj["frames"], str):
            ref_frames = self._load_frames_from_npz(ref_traj["frames"])
        else:
            ref_frames = ref_traj["frames"]

        if isinstance(traj_sim["frames"], str):
            traj_sim_frames = self._load_frames_from_npz(traj_sim["frames"])
        else:
            traj_sim_frames = traj_sim["frames"]

        if isinstance(traj_diff["frames"], str):
            traj_diff_frames = self._load_frames_from_npz(traj_diff["frames"])
        else:
            traj_diff_frames = traj_diff["frames"]

        # Apply uniform subsampling to get consistent frame counts
        num_frames_to_sample = getattr(self.config, "max_frames", 8)

        # For all trajectories, use uniform subsampling
        ref_frames, ref_indices = self._linspace_subsample_frames(ref_frames, num_frames_to_sample)
        ref_progress = [idx / (len(ref_frames) - 1) for idx in ref_indices]

        traj_sim_frames, traj_sim_indices = self._linspace_subsample_frames(traj_sim_frames, num_frames_to_sample)
        traj_sim_progress = [idx / (len(traj_sim_frames) - 1) for idx in traj_sim_indices]

        traj_diff_frames, traj_diff_indices = self._linspace_subsample_frames(traj_diff_frames, num_frames_to_sample)
        traj_diff_progress = [idx / (len(traj_diff_frames) - 1) for idx in traj_diff_indices]

        # Pad all trajectories to max_frames if needed
        ref_frames, ref_progress = self._pad_trajectory_to_max_frames(ref_frames, ref_progress, num_frames_to_sample)
        traj_sim_frames, traj_sim_progress = self._pad_trajectory_to_max_frames(
            traj_sim_frames, traj_sim_progress, num_frames_to_sample
        )
        traj_diff_frames, traj_diff_progress = self._pad_trajectory_to_max_frames(
            traj_diff_frames, traj_diff_progress, num_frames_to_sample
        )

        # Create SimilaritySample
        sample = SimilaritySample(
            reference_trajectory=Trajectory(
                frames=ref_frames,
                frames_shape=ref_frames.shape,
                id=ref_traj["id"],
                task=ref_traj["task"],
                lang_vector=ref_traj["lang_vector"],
                data_source=ref_traj["data_source"],
                quality_label=ref_traj.get("quality_label"),
                is_robot=ref_traj["is_robot"],
                target_progress=ref_progress,
                metadata={"data_gen_strategy": "optimal_similarity"},
            ),
            traj_sim_trajectory=Trajectory(
                frames=traj_sim_frames,
                frames_shape=traj_sim_frames.shape,
                id=traj_sim["id"],
                task=traj_sim["task"],
                lang_vector=traj_sim["lang_vector"],
                data_source=traj_sim["data_source"],
                quality_label=traj_sim["quality_label"],
                is_robot=traj_sim["is_robot"],
                target_progress=traj_sim_progress,
                metadata={"data_gen_strategy": "optimal_similarity"},
            ),
            traj_diff_trajectory=Trajectory(
                frames=traj_diff_frames,
                frames_shape=traj_diff_frames.shape,
                id=traj_diff["id"],
                task=traj_diff["task"],
                lang_vector=traj_diff["lang_vector"],
                data_source=traj_diff["data_source"],
                quality_label=traj_diff["quality_label"],
                is_robot=traj_diff["is_robot"],
                target_progress=traj_diff_progress,
                metadata={"data_gen_strategy": "optimal_similarity"},
            ),
            data_gen_strategy="optimal_similarity",
        )

        return sample
