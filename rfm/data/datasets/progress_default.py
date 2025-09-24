import numpy as np

from rfm.data.dataset_types import ProgressSample, Trajectory
from .base import RFMBaseDataset
from .helpers import linspace_subsample_frames, pad_trajectory_to_max_frames_np
from rfm.utils.distributed import rank_0_print


class ProgressDefaultDataset(RFMBaseDataset):
    """Dataset that generates progress samples by iterating through each trajectory in the dataset."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)
        self.current_idx = 0
        rank_0_print(f"ProgressDataset initialized with {len(self.robot_trajectories)} trajectories")

    def _create_progress_sample(self, idx: int) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        # Get the trajectory
        traj = self.dataset[idx]

        # Get frames
        frames = self._get_trajectory_frames(idx)

        # Use linspace sampling to get max_frames
        max_frames = self.config.max_frames
        frames, frame_indices = linspace_subsample_frames(frames, max_frames)

        # Calculate progress based on the sampled frame indices
        total_frames = len(self._get_trajectory_frames(idx))
        progress = [(idx + 1) / total_frames for idx in frame_indices]

        # Pad frames and progress if needed
        frames, progress = pad_trajectory_to_max_frames_np(frames, progress, max_frames)

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

    def __len__(self):
        return len(self.robot_trajectories)

    def __getitem__(self, idx):
        return self._create_progress_sample(idx)
