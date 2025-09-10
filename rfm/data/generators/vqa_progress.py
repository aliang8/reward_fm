from rfm.data.generators.base import BaseDataGenerator
from rfm.data.dataset_types import ProgressSample
from rfm.utils.logging import rank_0_print, timer
from rfm.data.generators.helpers import (
    pad_trajectory_to_max_frames,
    subsample_frames_and_progress,
    create_rewind_trajectory,
    load_frames_from_npz,
    DataGenStrat,
)
from enum import Enum
import random
from rfm.data.dataset_types import Trajectory


class VQAProgressGenerator(BaseDataGenerator):
    """Data generator for VQA progress samples."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)

    def __next__(self):
        return self._create_progress_sample()

    def _create_progress_sample(self):
        # Use preprocessed chosen trajectories from index maps
        if not self.optimal_by_task:
            raise ValueError("No chosen trajectories found for preference generation")

        # Get a random task and chosen trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))

        optimal_indices = self.optimal_by_task[task_name]
        while not optimal_indices:
            task_name = random.choice(list(self.optimal_by_task.keys()))
            optimal_indices = self.optimal_by_task[task_name]

        idx = random.choice(optimal_indices)
        traj = self.dataset[idx]

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
            frames = load_frames_from_npz(traj["frames"])

            # subsample frames and progress
            frames, progress, metadata = subsample_frames_and_progress(frames, self.config.max_frames)

            # pad frames and progress to max_frames
            frames, progress = pad_trajectory_to_max_frames(frames, progress, self.config.max_frames)

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
