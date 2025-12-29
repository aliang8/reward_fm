from typing import Dict, List, Any, Optional

import numpy as np
import torch
from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    load_embeddings_from_path,
    load_frames_from_npz,
    convert_absolute_to_relative_progress,
    create_trajectory_from_dict,
)
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

        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None

        # Use linspace sampling to get max_frames
        max_frames = self.config.max_frames

        # Load data (embeddings or frames)
        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]
            data = video_embeddings
            total_frames = video_embeddings.shape[0] if hasattr(video_embeddings, "shape") else len(video_embeddings)
            use_embeddings = True
        else:
            frames = load_frames_from_npz(traj["frames"])
            data = frames
            total_frames = len(frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        frames_shape_orig = data.shape

        # Compute progress based on type
        if self.config.progress_pred_type == "absolute_wrt_total_frames":
            progress_abs = [(idx + 1) / total_frames for idx in frame_indices]
        elif self.config.progress_pred_type.startswith("absolute"):
            # absolute_first_frame: use linspace logic
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]
        else:  # relative_first_frame
            # For relative, we still compute absolute first, then convert
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if self.config.progress_pred_type == "relative_first_frame":
            progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            progress = progress_abs

        metadata = {
            "quality_label": traj["quality_label"],
            "data_source": traj["data_source"],
            "task": traj["task"],
            "id": traj["id"],
            "video_path": video_path,
        }

        trajectory = create_trajectory_from_dict(
            traj,
            overrides={
                "frames": data if not use_embeddings else None,
                "frames_shape": frames_shape_orig,
                "video_embeddings": data if use_embeddings else None,
                "text_embedding": text_embedding,
                "lang_vector": np.array(traj["lang_vector"]),
                "target_progress": progress,
                "metadata": metadata,
            },
        )
        trajectory = self._post_process_trajectory(trajectory)

        # Create progress sample
        sample = ProgressSample(trajectory=trajectory)

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
