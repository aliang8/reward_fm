from rfm.data.datasets.base import RFMBaseDataset
from rfm.data.dataset_types import ProgressSample
from rfm.utils.logging import rank_0_print, timer
from rfm.data.datasets.helpers import (
    pad_trajectory_to_max_frames,
    subsample_frames_and_progress,
    create_rewind_trajectory,
    load_frames_from_npz,
    DataGenStrat,
)
from enum import Enum
import random
from rfm.data.dataset_types import Trajectory
from typing import Dict, Optional


class VQAProgressDataset(RFMBaseDataset):
    """Data generator for VQA progress samples."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)
        self.iter_dataset = self.dataset.filter(lambda x: x["quality_label"] not in ["failure", "suboptimal"])

    def __len__(self):
        return len(self.iter_dataset)

    def __getitem__(self, idx):
        """Iterate over one sample per trajectory in the dataset."""
        dataset_len = len(self.iter_dataset)
        traj = self.iter_dataset[idx % dataset_len]
        sample = self._create_progress_sample(traj)
        return sample

    def _create_progress_sample(self, traj: Dict):
        # either return the original trajectory, rewinded traj, or wrong task traj
        prob = random.random()
        if prob < 0.33:
            strategy = "default"
        elif prob < 0.66:
            strategy = DataGenStrat.REWIND_SAME_TASK
        else:
            strategy = DataGenStrat.DIFFERENT_TASK

        if strategy == DataGenStrat.REWIND_SAME_TASK:
            traj = create_rewind_trajectory(traj, max_frames=self.config.max_frames)
            frames = traj["frames"]
            progress = traj["target_progress"]
            metadata = traj["metadata"]
        else:
            if strategy == DataGenStrat.DIFFERENT_TASK:
                other_traj = self._create_different_task_trajectory(traj)
                if other_traj is None:
                    prob = random.random()
                    if prob < 0.5:
                        strategy = DataGenStrat.REWIND_SAME_TASK
                        other_traj = create_rewind_trajectory(traj, max_frames=self.config.max_frames)
                        traj = other_traj
                    else:
                        # nothing happens, we use the same trajectory
                        strategy = "default"
                else:
                    traj = other_traj

            if strategy == DataGenStrat.REWIND_SAME_TASK:
                frames = traj["frames"]
                progress = traj["target_progress"]
                metadata = traj["metadata"]
            else:
                frames = load_frames_from_npz(traj["frames"])

                # subsample frames and progress
                frames, progress, metadata = subsample_frames_and_progress(frames, self.config.max_frames)

                # pad frames and progress to max_frames
                frames, progress = pad_trajectory_to_max_frames(frames, progress, self.config.max_frames)

        if strategy == DataGenStrat.DIFFERENT_TASK:
            progress = [0.0] * len(progress)

        progress_traj = Trajectory(
            frames=frames,
            target_progress=progress,
            frames_shape=frames.shape,
            id=traj["id"],
            task=traj["task"],
            lang_vector=traj["lang_vector"],
            data_source=traj["data_source"],
            quality_label=traj["quality_label"],
            is_robot=traj["is_robot"],
            data_gen_strategy=strategy,
            metadata=metadata,
        )

        return ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
        )

    def _create_different_task_trajectory(self, chosen_traj: Dict) -> Optional[Dict]:
        """Create a trajectory from a different task than the chosen trajectory.

        This function tries to find trajectories from different tasks.
        Returns None if no other tasks are available.

        Args:
            chosen_traj: The chosen trajectory dictionary

        Returns:
            Optional[Dict]: The rejected trajectory, or None if none available
        """
        # Find other tasks
        other_tasks = [task for task in self.optimal_by_task.keys() if task != chosen_traj["task"]]
        if other_tasks:
            other_task = random.choice(other_tasks)
            other_task_indices = self.optimal_by_task[other_task]

            if other_task_indices:
                other_idx = random.choice(other_task_indices)
                other_traj = self.dataset[other_idx]

                # Check if it's not the same trajectory
                if other_traj["id"] != chosen_traj["id"]:
                    return other_traj
        else:
            # Only one task available
            return None

        return None
