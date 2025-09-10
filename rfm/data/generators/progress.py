from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.utils.logging import rank_0_print
from typing import Dict, List
from rfm.data.generators.base import BaseDataGenerator
from tqdm import tqdm
import numpy as np
from rfm.data.generators.helpers import linspace_subsample_frames, pad_trajectory_to_max_frames


class ProgressGenerator(BaseDataGenerator):
    """Dataset that generates progress samples by iterating through each trajectory in the dataset."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)
        self.current_idx = 0
        rank_0_print(f"ProgressGenerator initialized with {len(self.robot_trajectories)} trajectories")

    def _create_progress_sample(self, traj_idx: int) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        # Get the trajectory
        traj = self.dataset[traj_idx]

        # Get frames
        frames = self._get_trajectory_frames(traj_idx)

        # Use linspace sampling to get max_frames
        max_frames = self.config.max_frames
        frames, frame_indices = linspace_subsample_frames(frames, max_frames)

        # Calculate progress based on the sampled frame indices
        total_frames = len(self._get_trajectory_frames(traj_idx))
        progress = [idx / (total_frames - 1) if total_frames > 1 else 0.0 for idx in frame_indices]

        # Pad frames and progress if needed
        frames, progress = pad_trajectory_to_max_frames(frames, progress, max_frames)

        metadata = {
            "quality_label": traj["quality_label"],
            "data_source": traj["data_source"],
            "task": traj["task"],
        }

        # Create trajectory for the progress sample
        trajectory = Trajectory(
            frames=frames,
            frames_shape=frames.shape if hasattr(frames, "shape") else (len(frames),),
            id=traj["id"],
            task=traj["task"],
            lang_vector=np.array(traj["lang_vector"]),
            data_source=traj["data_source"],
            quality_label=traj["quality_label"],
            is_robot=traj["is_robot"],
            target_progress=progress,
            metadata=metadata,
        )

        # Create progress sample
        sample = ProgressSample(trajectory=trajectory)

        return sample

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        """Get the next sample by iterating through trajectories."""
        if self.current_idx >= len(self.robot_trajectories):
            raise StopIteration

        # Get the current trajectory index
        traj_idx = self.robot_trajectories[self.current_idx]

        # Generate the progress sample
        sample = self._create_progress_sample(traj_idx)

        self.current_idx += 1
        return sample

    def __len__(self):
        return len(self.robot_trajectories)
