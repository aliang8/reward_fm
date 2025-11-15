import numpy as np
import torch
from tqdm import tqdm

from rfm.data.dataset_types import PreferenceSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    subsample_segment_frames,
    compute_progress_from_segment,
    load_frames_from_npz,
)
from rfm.utils.distributed import rank_0_print


class PairedSuccessFailureSampler(RFMBaseSampler):
    """Dataset that generates preference samples by pairing successful and failed trajectories for the same task."""

    def __init__(self, config, dataset, combined_indices, dataset_success_cutoff_map=None, is_evaluation=False, verbose=True, **kwargs):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(f"Generated {len(self.sample_indices)} success-failure sample indices")

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible success-failure sample indices (not the actual samples)."""
        sample_indices = []

        # Group trajectories by task and success status
        task_success_trajs = {}
        task_failure_trajs = {}

        print(f"Generating success-failure samples for {len(self.robot_trajectories)} trajectories")

        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            task = traj["task"]
            quality_label = traj["quality_label"]

            if task not in task_success_trajs:
                task_success_trajs[task] = []
                task_failure_trajs[task] = []

            if quality_label == "successful":
                task_success_trajs[task].append(traj_idx)
            elif quality_label == "failure":
                task_failure_trajs[task].append(traj_idx)

        print(f"Generated {len(task_success_trajs)} success tasks and {len(task_failure_trajs)} failure tasks")

        # Generate all possible pairs
        for task in tqdm(task_success_trajs, desc="Generating success-failure samples"):
            success_indices = task_success_trajs[task]
            failure_indices = task_failure_trajs[task]

            if not success_indices or not failure_indices:
                continue

            # Create all possible pairs (successful is chosen, failed is rejected)
            for success_idx in success_indices:
                for failure_idx in failure_indices:
                    # Store just the indices needed to generate the sample later
                    sample_indices.append({
                        "success_traj_idx": success_idx,
                        "failure_traj_idx": failure_idx,
                        "task": task,
                    })

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        success_idx = sample_idx_info["success_traj_idx"]
        failure_idx = sample_idx_info["failure_traj_idx"]

        # Get the trajectories
        success_traj = self.dataset[success_idx]
        failure_traj = self.dataset[failure_idx]

        # Get frames
        success_frames = load_frames_from_npz(success_traj["frames"])
        failure_frames = load_frames_from_npz(failure_traj["frames"])

        # Subsample frames
        # Get success cutoff from pre-loaded map for both trajectories
        ds_key_success = success_traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key_success, self.config.max_success)

        ds_key_failure = failure_traj["data_source"]
        failure_cutoff = self.dataset_success_cutoff_map.get(ds_key_failure, self.config.max_success)

        subsampled_success, start_idx_success, end_idx_success, indices_success = subsample_segment_frames(
            success_frames, self.config.max_frames
        )
        success_progress = compute_progress_from_segment(
            num_frames_total=success_frames.shape[0],
            start_idx=start_idx_success,
            end_idx=end_idx_success,
            frame_indices=indices_success,
            progress_pred_type=self.config.progress_pred_type,
            success_cutoff=success_cutoff,
        )
        success_metadata = {
            "start_idx": start_idx_success,
            "end_idx": end_idx_success,
            "subsampled_indices": indices_success,
        }
        success_frames = subsampled_success

        subsampled_failure, start_idx_failure, end_idx_failure, indices_failure = subsample_segment_frames(
            failure_frames, self.config.max_frames
        )
        failure_progress = compute_progress_from_segment(
            num_frames_total=failure_frames.shape[0],
            start_idx=start_idx_failure,
            end_idx=end_idx_failure,
            frame_indices=indices_failure,
            progress_pred_type=self.config.progress_pred_type,
            success_cutoff=failure_cutoff,
        )
        failure_metadata = {
            "start_idx": start_idx_failure,
            "end_idx": end_idx_failure,
            "subsampled_indices": indices_failure,
        }
        failure_frames = subsampled_failure

        chosen_trajectory = Trajectory(
            frames=success_frames,
            frames_shape=success_traj["frames_shape"],
            id=success_traj["id"],
            task=success_traj["task"],
            lang_vector=np.array(success_traj["lang_vector"]),
            data_source=success_traj["data_source"],
            quality_label=success_traj["quality_label"],
            is_robot=success_traj["is_robot"],
            target_progress=success_progress,
            metadata=success_metadata,
        )

        rejected_trajectory = Trajectory(
            frames=failure_frames,
            frames_shape=failure_traj["frames_shape"],
            id=failure_traj["id"],
            task=failure_traj["task"],
            lang_vector=np.array(failure_traj["lang_vector"]),
            data_source=failure_traj["data_source"],
            quality_label=failure_traj["quality_label"],
            is_robot=failure_traj["is_robot"],
            target_progress=failure_progress,
            metadata=failure_metadata,
        )

        # Create preference sample (successful is chosen, failed is rejected)
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy="success_failure",
        )

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
