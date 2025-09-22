import numpy as np
from tqdm import tqdm

from rfm.data.dataset_types import PreferenceSample, Trajectory
from .base import RFMBaseDataset
from .helpers import subsample_frames_and_progress
from rfm.utils.logging import rank_0_print


class PairedSuccessFailureDataset(RFMBaseDataset):
    """Dataset that generates preference samples by pairing successful and failed trajectories for the same task."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)

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
        success_frames = self._get_trajectory_frames(success_idx)
        failure_frames = self._get_trajectory_frames(failure_idx)

        # Subsample frames
        success_frames, success_progress, success_metadata = subsample_frames_and_progress(
            success_frames, max_frames=self.config.max_frames
        )
        failure_frames, failure_progress, failure_metadata = subsample_frames_and_progress(
            failure_frames, max_frames=self.config.max_frames
        )

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

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])

    def __len__(self):
        return len(self.sample_indices)
