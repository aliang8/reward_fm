#!/usr/bin/env python3
"""
Data generator for reward alignment evaluation.

This generator creates subsequence samples from trajectories for progress prediction evaluation.
For each trajectory, it creates multiple subsequences (0:2, 0:4, 0:6, etc.) and formats them
as PreferenceSample objects that can be evaluated by the model.
"""

from tqdm import tqdm

from rfm.data.dataset_types import ProgressSample, Trajectory
from .base import RFMBaseDataset
from .helpers import linspace_subsample_frames, pad_trajectory_to_max_frames_np
from rfm.utils.distributed import rank_0_print


class RewardAlignmentDataset(RFMBaseDataset):
    """
    Data generator that creates subsequence samples for reward alignment evaluation.

    For each trajectory, creates subsequences of frames (0:2, 0:4, 0:6, etc.)
    and formats them as PreferenceSample objects for evaluation.
    """

    def __init__(
        self, config, is_evaluation=False, verbose=True, max_trajectories: int | None = None, frame_step: int = 2
    ):
        super().__init__(config, is_evaluation, verbose=verbose)

        self.max_trajectories = config.max_trajectories
        self.frame_step = frame_step
        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} reward alignment sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories"
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible subsequence sample indices (not the actual samples)."""
        sample_indices = []

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None:
            trajectories_to_process = self.robot_trajectories[: self.max_trajectories]

        print(f"Generating subsequence samples for {len(trajectories_to_process)} trajectories")

        for traj_idx in tqdm(trajectories_to_process, desc="Generating subsequence indices"):
            traj = self.dataset[traj_idx]

            # Get trajectory length from frames
            frames_path = traj.get("frames")
            if not frames_path:
                continue

            # Get frames and determine number of frames
            frames = self._get_trajectory_frames(traj_idx)
            if frames is None or len(frames) < self.frame_step:
                continue
            num_frames = len(frames)

            # Create subsequence indices: 0:2, 0:4, 0:6, etc.
            for end_idx in range(self.frame_step, num_frames + 1, self.frame_step):
                sample_indices.append({"traj_idx": traj_idx, "end_idx": end_idx, "num_frames": num_frames})

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single subsequence sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        end_idx = sample_idx_info["end_idx"]
        num_frames = sample_idx_info["num_frames"]

        # Get the original trajectory
        original_traj = self.dataset[traj_idx]

        # Get frames and create subsequence
        frames = self._get_trajectory_frames(traj_idx)
        if frames is None or len(frames) == 0:
            return None

        # Create subsequence frames
        subsequence_frames = frames[:end_idx]

        # Get max_frames from config
        max_frames = self.config.max_frames

        # Uniform subsample to max_frames
        subsequence_frames, _ = linspace_subsample_frames(subsequence_frames, max_frames)

        # Use the existing helper function to pad/subsample frames
        padded_frames, _ = pad_trajectory_to_max_frames_np(subsequence_frames, [0], max_frames)

        # Ground truth progress: linear from 0 to 1
        gt_progress = end_idx / num_frames

        # Create metadata for the subsequence
        metadata = {
            "subsequence_end": end_idx,
            "ground_truth_progress": gt_progress,
            "data_gen_strategy": "reward_alignment",
        }

        # Create trajectory for the subsequence
        subsequence_trajectory = Trajectory(
            id=original_traj["id"],
            task=original_traj["task"],
            frames=padded_frames,
            frames_shape=padded_frames.shape,
            data_source=original_traj["data_source"],
            lang_vector=original_traj["lang_vector"],
            is_robot=original_traj["is_robot"],
            quality_label=original_traj["quality_label"],
            data_gen_strategy="reward_alignment",
            target_progress=[gt_progress],
            metadata=metadata,
        )

        sample = ProgressSample(trajectory=subsequence_trajectory, sample_type="progress")

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
