from typing import Dict, List, Any, Optional

import numpy as np
import torch
from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.utils.distributed import rank_0_print


class ProgressDefaultSampler(RFMBaseSampler):
    """Dataset that generates progress samples by iterating through each trajectory in the dataset, used in policy ranking."""

    def __init__(
        self,
        max_trajectories: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_trajectories = max_trajectories
        rank_0_print(
            f"ProgressDefaultSampler initialized with {len(self.robot_trajectories)} trajectories", verbose=self.verbose
        )

        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(f"Generated {len(self.sample_indices)} sample indices", verbose=self.verbose)

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate all possible sample indices."""
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None and self.max_trajectories < len(self.robot_trajectories):
            trajectories_to_process = self._local_random.sample(self.robot_trajectories, self.max_trajectories)

        rank_0_print(
            f"Generating progress default samples for {len(trajectories_to_process)} trajectories", verbose=self.verbose
        )

        sample_indices = []
        for i in trajectories_to_process:
            sample_indices.append({"traj_idx": i, "video_path": self.dataset[i]["frames"], "id": self.dataset[i]["id"]})
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        traj_idx = sample_idx_info["traj_idx"]
        video_path = sample_idx_info["video_path"]

        traj = self.dataset[traj_idx]

        metadata = {
            "quality_label": traj["quality_label"],
            "data_source": traj["data_source"],
            "task": traj["task"],
            "id": traj["id"],
            "video_path": video_path,
        }

        trajectory = self._get_traj_from_data(
            traj=traj,
            metadata=metadata,
        )

        # Create progress sample
        sample = ProgressSample(trajectory=trajectory)

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
