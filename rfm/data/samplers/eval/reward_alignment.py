#!/usr/bin/env python3
"""
Data generator for reward alignment evaluation.

This generator creates subsequence samples from trajectories for progress prediction evaluation.
For each trajectory, it creates multiple subsequences (0:2, 0:4, 0:6, etc.) and formats them
as PreferenceSample objects that can be evaluated by the model.
"""

import random
import torch
from tqdm import tqdm

from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
    load_frames_from_npz,
    compute_success_labels,
)
from rfm.utils.distributed import rank_0_print


class RewardAlignmentSampler(RFMBaseSampler):
    """
    Data generator that creates subsequence samples for reward alignment evaluation.

    For each trajectory, creates subsequences of frames (0:2, 0:4, 0:6, etc.)
    and formats them as PreferenceSample objects for evaluation.
    """

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        max_trajectories: int | None = None,
        frame_step: int = 1,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        self.max_trajectories = max_trajectories
        self.frame_step = frame_step
        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} reward alignment sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories",
            verbose=self.verbose,
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible subsequence sample indices (not the actual samples)."""
        sample_indices = []

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None and self.max_trajectories < len(self.robot_trajectories):
            trajectories_to_process = random.sample(self.robot_trajectories, self.max_trajectories)

        rank_0_print(
            f"Generating subsequence samples for {len(trajectories_to_process)} trajectories", verbose=self.verbose
        )

        for traj_idx in trajectories_to_process:
            traj = self.dataset[traj_idx]
            num_frames = traj["num_frames"]
            # Create subsequence indices: 0:1, 0:2, 0:3, etc.
            for end_idx in range(self.frame_step, num_frames + 1, self.frame_step):
                sample_indices.append({
                    "traj_idx": traj_idx,
                    "end_idx": end_idx,
                    "num_frames": num_frames,
                    "video_path": traj["frames"],
                    "id": traj["id"],
                })

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single subsequence sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        end_idx = sample_idx_info["end_idx"]
        num_frames = sample_idx_info["num_frames"]

        # Get the original trajectory
        original_traj = self.dataset[traj_idx]

        # Get frames and create subsequence
        frames = None
        video_embeddings = None
        text_embedding = None

        # Ground truth progress: linear from 0 to 1
        # If starts with "absolute", use linspace logic; if "relative", use 1/num_frames
        if self.config.progress_pred_type.startswith("absolute"):
            gt_progress = (end_idx - 1) / (num_frames - 1)
        else:  # relative_first_frame
            gt_progress = 1 / num_frames

        if self.config.load_embeddings and original_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(original_traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]

            video_embeddings = video_embeddings[:end_idx]

            subsequence_video_embeddings, frame_indices = linspace_subsample_frames(
                video_embeddings, self.config.max_frames
            )
            frames_shape_orig = subsequence_video_embeddings.shape
        else:
            frames = load_frames_from_npz(original_traj["frames"])
            if frames is None or len(frames) == 0:
                return None

            # Create subsequence frames
            subsequence_frames = frames[:end_idx]

            # Get max_frames from config
            max_frames = self.config.max_frames

            # Uniform subsample to max_frames
            subsequence_frames, frame_indices = linspace_subsample_frames(subsequence_frames, max_frames)
            frames_shape_orig = subsequence_frames.shape

        # Create progress values for each subsampled frame
        # Progress should linearly interpolate from 0 to gt_progress across the frames
        num_subsampled = len(frame_indices)
        if num_subsampled > 1:
            # Linear interpolation from 0 to gt_progress
            progress_values = [gt_progress * (idx / (num_subsampled - 1)) for idx in range(num_subsampled)]
        else:
            progress_values = [gt_progress]

        # Pad trajectory and progress
        if self.config.load_embeddings and original_traj.get("embeddings_path"):
            video_embeddings, padded_progress = pad_trajectory_to_max_frames_torch(
                subsequence_video_embeddings, progress_values, self.config.max_frames
            )
        else:
            frames, padded_progress = pad_trajectory_to_max_frames_np(subsequence_frames, progress_values, max_frames)

        # Create metadata for the subsequence
        metadata = {
            "subsequence_end": end_idx,
            "ground_truth_progress": gt_progress,
            "data_gen_strategy": "reward_alignment",
            "id": original_traj["id"],
            "video_path": sample_idx_info["video_path"],
        }

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=padded_progress,
            data_source=original_traj["data_source"],
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )

        # Create trajectory for the subsequence
        subsequence_trajectory = Trajectory(
            id=original_traj["id"],
            task=original_traj["task"],
            frames=frames,
            frames_shape=frames_shape_orig,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            data_source=original_traj["data_source"],
            lang_vector=original_traj["lang_vector"],
            is_robot=original_traj["is_robot"],
            quality_label=original_traj["quality_label"],
            data_gen_strategy="reward_alignment",
            target_progress=padded_progress,
            partial_success=original_traj.get("partial_success"),
            success_label=success_label,
            metadata=metadata,
        )

        sample = ProgressSample(trajectory=subsequence_trajectory, sample_type="progress")

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
