import numpy as np

from rfm.data.dataset_types import ProgressSample, Trajectory
from .base import RFMBaseDataset
from .helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
)
from rfm.utils.distributed import rank_0_print


class ProgressDefaultDataset(RFMBaseDataset):
    """Dataset that generates progress samples by iterating through each trajectory in the dataset."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)
        self.current_idx = 0
        rank_0_print(f"ProgressDataset initialized with {len(self.robot_trajectories)} trajectories")
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(f"Generated {len(self.sample_indices)} sample indices")

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible sample indices."""
        return [
            {"idx": i, "video_path": self.dataset[i]["frames"], "id": self.dataset[i]["id"]}
            for i in range(len(self.robot_trajectories))
        ]

    def _create_progress_sample(self, idx: int) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        sample_info = self.sample_indices[idx]

        # Get the trajectory
        traj = self.dataset[sample_info["idx"]]

        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None

        # Use linspace sampling to get max_frames
        max_frames = self.config.max_frames

        if self.config.load_embeddings and traj.get("embeddings_path"):
            # Load embeddings from path
            video_embeddings = load_embeddings_from_path(traj["embeddings_path"], "video_embeddings")
            text_embedding = load_embeddings_from_path(traj["embeddings_path"], "text_embedding")

            # Subsample video embeddings
            video_embeddings, frame_indices = linspace_subsample_frames(video_embeddings, max_frames)

            # Calculate progress based on the sampled frame indices
            total_frames = video_embeddings.shape[0] if hasattr(video_embeddings, "shape") else len(video_embeddings)
            progress = [(idx + 1) / total_frames for idx in frame_indices]

            # Pad embeddings and progress if needed
            video_embeddings, progress = pad_trajectory_to_max_frames_torch(video_embeddings, progress, max_frames)
        else:
            # Get frames
            frames = self._get_trajectory_frames(idx)

            # Subsample frames
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
            "id": sample_info["id"],
            "video_path": sample_info["video_path"],
        }

        # Create trajectory for the progress sample
        trajectory = Trajectory(
            frames=frames,
            frames_shape=frames.shape if frames is not None and hasattr(frames, "shape") else None,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
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
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._create_progress_sample(idx)
