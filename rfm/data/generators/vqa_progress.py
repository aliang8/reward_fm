from rfm.data.generators.base import BaseDataGenerator
from rfm.data.dataset_types import ProgressSample
from rfm.utils.logging import rank_0_print, timer
from rfm.data.generators.helpers import (
    linspace_subsample_frames,
    randomly_subsample_frames,
    pad_trajectory_to_max_frames,
    subsample_frames_and_progress,
    create_rewind_trajectory,
)
from enum import Enum
import random


class VQAProgressGenerator(BaseDataGenerator):
    """Data generator for VQA progress samples."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        super().__init__(config, is_evaluation, verbose=verbose)

    def __next__(self):
        return self.progress_generator.__next__()

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

        chosen_idx = random.choice(optimal_indices)
        chosen_traj = self.dataset[chosen_idx]

        pass
